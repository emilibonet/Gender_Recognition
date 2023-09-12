import argparse

import os
import cv2
import shutil
import json
import random
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader

from initial_design import BaselineBodyPredictor, MultiheadBodyPredictor, FaceDetector, CustomDataset

import wandb


def create_partitions(root:str, splits=[0.7, 0.15, 0.15], seed:int=1234):
    # Collect the samples
    samples = []
    for pathset in [f"{root}/{p}/" for p in os.listdir(root)]:
        if not os.path.exists(pathset+"annotations/curated.json"):
            continue
        with open(pathset+"annotations/curated.json", "r") as curated:
            ann = json.load(curated)
            for imgpath in ann:
                for sample in ann[imgpath]:
                    samples.append({'path':imgpath, 'bbb':sample['bbb'], 'fbb':sample['fbb'], 'gender':sample['gender'], 'age':sample['age']})
    samples = np.array(samples)

    # Partition the samples
    n = len(samples)
    print(f"Total number of samples for training: {n}")
    mask = np.zeros(n, dtype=int)
    tr_limit = round(n*splits[0])
    va_limit = round(n*(splits[0]+splits[1]))
    print(f"Samples for training split: {tr_limit} ({splits[0]*100}%)")
    print(f"Samples for validation split: {va_limit-tr_limit} ({splits[1]*100}%)")
    print(f"Samples for testing split: {n-va_limit} ({splits[2]*100}%)")
    mask[tr_limit:va_limit], mask[va_limit:] = 1, 2
    random.seed(seed)
    random.shuffle(mask)
    return samples[mask==0], samples[mask==1], samples[mask==2]


def blur_faces(img, kernel:int=61):
    multiple = True
    if type(img) is not list:
        multiple = False
        img = [img]
    facedet = FaceDetector()
    res = []
    for im in img:
        if type(im) is str:
            im = plt.imread(im)
        istensor = False
        if type(im) is torch.Tensor:
            istensor = True
            im = im.permute(1,2,0).numpy()*255
            im = im.astype(np.uint8)
        blurred_im = cv2.GaussianBlur(im, (kernel, kernel), 0)
        mask = np.zeros((*im.shape[:2], 3))
        for _,bbox in facedet.predict(im):
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = 1
        im = np.where(mask==np.array([1, 1, 1]), blurred_im, im)
        if istensor:
            im = transforms.ToTensor()(im)
        if not multiple:
            return im
        res.append(im)
    return res

class FacialBlur(object):
    """Implements a custom transformation for the facial blur."""
    def __init__(self, kernel:int=61, prob:float=1.0):
        """Defines the facial blur transformation object based on the kernel (odd integer: the higher, the more blur) and a probability to apply this transformation."""
        self.kernel = kernel
        self.prob = prob

    def __call__(self, img):
        if type(img) is not torch.Tensor:
            raise ValueError(f"Image is of type {type(img)}; expected type {torch.Tensor}.")
        if random.uniform(0,1) < self.prob:
            return blur_faces(img, self.kernel)
        return img
    
    def __repr__(self):
        return self.__class__.__name__ + f'(kernel={self.kernel}, prob={self.prob})'


def xvalid_splits(sequences, fold, num_valid_seq, config):
    """Defines the data loaders with the split corresponding to the fold.

    ---
    Arguments:
        - sequences: list of strings with the names of the MOT sequences that need to be considered for the splits.
        - fold: number of the current fold, which will define what sequences are using in the training and validation splits.
        - num_valid_seq: integer representing the number of validation sequences that need to be used.
        - config: configuration dictionary with all the relevant information (in this case, a field 'batch_size').
    
    Returns:
        - position 0: dictionary with keys 'train' containing the training data loader and 'valid' containing the validation data loader.
        - position 1: integer with the total number of samples in both splits.
        - position 2: integer with the number of samples in training split.
        - position 3: integer with the number of samples in validation split.
    """
    root = os.getcwd().split("cv-gender-age-recognition")[0]+"cv-gender-age-recognition/data/body"
    train_mask = np.ones(len(sequences), dtype=bool)
    train_mask[fold*num_valid_seq:(fold+1)*num_valid_seq] = False
    splits = {'train': sequences[train_mask],
                'valid': sequences[~train_mask]}
    tr_samples, va_samples = [], []
    for seq in os.listdir(root):
        curated_path = os.path.join(root, seq, 'annotations/curated.json')
        if (seq not in splits['train'] and seq not in splits['valid']) or not os.path.exists(curated_path):
            continue
        samples = []
        with open(curated_path) as curated:
            ann = json.load(curated)
            for imgpath in ann:
                for sample in ann[imgpath]:
                    samples.append({'path':imgpath, 'bbb':sample['bbb'], 'fbb':sample['fbb'], 'gender':sample['gender'], 'age':sample['age']})
        if seq in splits['train']:
            tr_samples.extend(samples)
        if seq in splits['valid']:
            va_samples.extend(samples)
    ntotal = len(tr_samples) + len(va_samples)
    print('Fold {}/{} - Training: {} ({:.2f}%); Validation: {} ({:.2f}%)'.format(fold+1, len(sequences)//num_valid_seq, len(tr_samples), len(tr_samples)/ntotal*100, len(va_samples), len(va_samples)/ntotal*100, 2))
    tform = transforms.Compose([
        transforms.ColorJitter(brightness=.5, hue=0.05),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.5),
        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.RandomHorizontalFlip(p=0.5)
    ])
    tr_dset, va_dset = CustomDataset(tr_samples, transform=tform, prob_tform=0.8), CustomDataset(va_samples, prob_tform=0)
    dloaders = {'train': DataLoader(tr_dset, batch_size=config['batch_size'], shuffle=True, num_workers=1, drop_last=True),
                'valid': DataLoader(va_dset, batch_size=config['batch_size'], shuffle=True, num_workers=1, drop_last=True)}
    return dloaders, ntotal, len(tr_samples), len(va_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Determines which type of architecture will be trained.", default="baseline", choices=["baseline", "multihead"])
    parser.add_argument("--backbone_model", help="Determines the backbone.", default="efficientnet", choices=["convnext", "resnet", "efficientnet"])
    parser.add_argument("--backbone_size", help="Determines the size of backbone.", default="small", choices=["tiny", "small", "base", "large"])
    parser.add_argument("--pretrained", help="If flagged, use backbone pretrained on ImageNet.", action='store_true')
    parser.add_argument("--cuda", help="If flagged, use cuda.", action='store_true')
    parser.add_argument("--epochs", help="Number of epochs with which the model will be trained.", default=30, type=int, choices=range(1,101))
    parser.add_argument("--save_as", help="Name the execution to save statistics and trained model.", default=None, type=str)
    parser.add_argument("--wandb", help="Set entity name for logging training statistics. If left unused, statistics are not logged with W&B.", default=None, type=str)
    parser.add_argument("--wandb_key", help="Authorization token for your user. --wandb flag required.", default=None, type=str)

    args = parser.parse_args()
    root = os.getcwd().split("cv-gender-age-recognition")[0]+"cv-gender-age-recognition/data/body"

    if args.cuda and not torch.cuda.is_available():
        print("Cuda is not available; using CPU instead.")
        args.cuda = False

    config = {
        'device':torch.device('cuda:0' if args.cuda else 'cpu'),
        'dropout':0.15,
        'lr':1e-2,
        'epochs':args.epochs,
        'momentum':0.75,
        'batch_size':14,
        'weight_decay':1e-3
    }

    # Set up W&B
    run = None
    if args.wandb is not None:
        if args.wandb_key is None:
            raise ValueError("W&B was flagged for use but no token was provided. Please insert your token with the command line flagg '--wandb_key'.")
        os.environ['WANDB_API_KEY'] = args.wandb_key

    # Set up cross validation
    tr_sequences = np.array(["MOT17-02", "MOT17-05", "MOT17-06", "MOT17-07", "MOT17-08", "MOT17-10", "MOT17-11", "MOT17-12"])
    te_sequences = np.array(["MOT17-09"])
    num_valid_seq = 2
    
    # Set up result saving: the parameters for the best performing epoch of each fold will be saved here along with the optimizer and the evaluation metrics.
    savedir = None
    if args.save_as is not None:
        savedir = os.getcwd().split("cv-gender-age-recognition")[0]+"cv-gender-age-recognition/trained_models/body/"+f"exp_{args.save_as}/"
        if os.path.isdir(savedir):
            shutil.rmtree(savedir)
        os.mkdir(savedir)
    

    # Cross Validation
    fold_stats = []
    for fold in range(len(tr_sequences)//num_valid_seq):
        # Prepare model
        if args.model == "baseline":
            model = BaselineBodyPredictor(backbone_model=args.backbone_model, backbone_size=args.backbone_size, pretrained_backbone=args.pretrained, dropout=config['dropout'])
            optimizer = {
                'age': optim.SGD(model.age_model.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay']),
                'gender': optim.SGD(model.gender_model.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
            }
        else:
            model = MultiheadBodyPredictor(backbone_model=args.backbone_model, backbone_size=args.backbone_size, pretrained_backbone=args.pretrained, dropout=config['dropout'])
            optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
        model.to(config['device'])

        # Train fold
        dloaders, ntotal, ntr, nva = xvalid_splits(tr_sequences, fold, num_valid_seq, config)
        if args.wandb is not None:
            run = wandb.init(project="gar_itw", config=config, reinit=True)
            run_name = args.save_as if args.save_as is not None else f"{args.model}_{args.backbone_model}_{args.backbone_size}"
            wandb.run.name = run_name + f"_fold_{fold}"
        optimal_checkpoint = os.path.join(savedir, f"state_dict_fold{fold}.pth") if savedir is not None else None
        stats = model.train_model(dloaders, optimizer, config, prints=True, ocp=optimal_checkpoint, wb_run=run)
        fold_stats.append({'stats':stats, 'ntotal':ntotal, 'ntr':ntr, 'nva':nva})
    if args.save_as is not None:
        with open(os.path.join(savedir, "statistics.json"), "w") as stats_file:
            json.dump(stats, stats_file, indent=4)
