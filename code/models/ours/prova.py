import os
import json
import copy
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import models, transforms

from initial_design import CustomDataset


class GenderBodyPredictor(nn.Module):
    def __init__(self, backbone:str="efficientnet", dropout=0):
        super(GenderBodyPredictor,self).__init__()
        self.dropout = dropout
        self.best_acc = 0
        if backbone == "efficientnet":
            self.backbone = models.efficientnet_b7(pretrained=True)
        elif backbone == "convnext":
            self.backbone = models.convnext_tiny(pretrained=True)
        self.mlp = nn.Sequential(
            nn.Linear(1000, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout, inplace=True),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout, inplace=True),

            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout, inplace=True),
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.eval()
    
    def forward(self, x):
        return self.mlp(self.backbone(x))


def xvalid_splits(sequences, fold, num_valid_seq, config):
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
        print('Fold {}/{} - Training: {} ({:.2f}%); Validation: {} {:.2f}%'.format(fold+1, len(sequences)//num_valid_seq, len(tr_samples), len(tr_samples)/ntotal*100, len(va_samples), len(va_samples)/ntotal*100, 2))
        tform = transforms.Compose([
            transforms.ColorJitter(brightness=.5, hue=0.05),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.5),
            transforms.RandomRotation(degrees=(-10, 10)),
            transforms.RandomHorizontalFlip(p=0.5)
        ])
        tr_dset, va_dset = CustomDataset(tr_samples), CustomDataset(va_samples)
        dloaders = {'train': DataLoader(tr_dset, batch_size=config['batch_size'], shuffle=True, num_workers=2, drop_last=True),
                    'valid': DataLoader(va_dset, batch_size=config['batch_size'], shuffle=True, num_workers=2, drop_last=True)}
        return dloaders, ntotal, len(tr_samples), len(va_samples)

if __name__ == "__main__":
    num_valid_seq = 2
    sequences = np.array(["MOT17-02", "MOT17-05", "MOT17-06", "MOT17-07", "MOT17-08", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-12"])
    config = {
        'device':torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        'dropout':0,
        'lr':1e-2,
        'epochs':100,
        'momentum':0.9,
        'batch_size':14
    }

    for fold in range(len(sequences)//num_valid_seq):
        # Prepare model and optimizer
        # model = models.efficientnet_b7(pretrained=True)
        # model.classifier[1] = nn.Sequential(
        #     nn.Linear(2560,1),
        #     nn.Sigmoid())
        # model = nn.Sequential(
        #     models.resnet18(pretrained=True),
        #     nn.Linear(1000,1),
        #     nn.Sigmoid())
        model = GenderBodyPredictor()
        # optimizer = optim.AdamW(model.parameters(), lr=config['lr'], amsgrad=True, weight_decay=config['weight_decay'])
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
        model.to(config['device'])
        model.train()


        # Train fold
        dloaders, ntotal, ntr, nva = xvalid_splits(sequences, fold, num_valid_seq, config)
        # x, y = next(iter(dloaders['train']))
        # print(y[0])
        gender_loss_fn = nn.BCELoss()
        for epoch in range(config['epochs']):
            print(f"Epoch {epoch+1}/{config['epochs']}")
            accum_gender_loss = accum_gender_acc = 0
            for x, y in dloaders['train']:
                x, y = x.to(config['device']), [label.to(config['device']) for label in y]
                optimizer.zero_grad()
                gout = model.forward(x)
                gloss = gender_loss_fn(gout.squeeze(1), y[0].to(torch.float))
                gloss.backward()
                optimizer.step()
                # Training statistics
                gpreds = torch.round(gout).detach()
                accum_gender_acc += torch.sum(gpreds == y[0].unsqueeze(1)).item()
                accum_gender_loss += gloss.item() * x.size(0)
            n = len(dloaders['train'].dataset)
            epoch_gender_loss, epoch_gender_acc = accum_gender_loss/n, accum_gender_acc/n
            print('GLoss: {:.4f}, GAcc: {:.4f}'.format(epoch_gender_loss, epoch_gender_acc))