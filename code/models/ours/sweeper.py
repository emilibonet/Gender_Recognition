import yaml
import argparse

import os
import numpy as np

import torch
from torch import optim

from initial_design import BaselineBodyPredictor, MultiheadBodyPredictor
from train import xvalid_splits

import wandb


def sweep_wrapper():
    global args
    with wandb.init(project="gar_itw", reinit=True) as run:
        config = wandb.config
        config['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        config['epochs'] = args.epochs

        # Prepare the data
        tr_sequences = np.array(["MOT17-02", "MOT17-05", "MOT17-06", "MOT17-07", "MOT17-08", "MOT17-10", "MOT17-11", "MOT17-12"])
        dloaders, _, _, _ = xvalid_splits(tr_sequences, 0, 2, config)

        # Prepare the model
        if args.model == "baseline":
            model = BaselineBodyPredictor(backbone_model=args.backbone_model, backbone_size=args.backbone_size, pretrained_backbone=True, dropout=config['dropout'])
            optimizer = {
                'age': optim.SGD(model.age_model.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay']),
                'gender': optim.SGD(model.gender_model.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
            }
        else:
            model = MultiheadBodyPredictor(backbone_model=args.backbone_model, backbone_size=args.backbone_size, pretrained_backbone=True, dropout=config['dropout'])
            optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
        model.to(config['device'])

        # Train & Validate
        stats = model.train_model(dloaders, optimizer, config, prints=True, ocp=None, wb_run=run)
        try:
            wandb.log({"max_valid_age_acc": max(stats['valid']['age']['accs']), "max_valid_gender_acc": max(stats['valid']['gender']['accs'])})
        except:
            print("Failed logging outside.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="Determines which type of architecture will be trained.", default='baseline', choices=['baseline', 'multihead'])
    parser.add_argument('--backbone_model', help="Determines the backbone.", default='efficientnet', choices=['convnext', 'resnet', 'efficientnet'])
    parser.add_argument('--backbone_size', help="Determines the size of backbone.", default='tiny', choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--epochs', help="Number of epochs with which the model will be trained.", default=30, type=int, choices=range(1,101))
    parser.add_argument('--wandb', help="Set entity name for logging training statistics. If not used, statistics are not logged with W&B.", default=None, type=str)
    parser.add_argument('--wandb_key', help="Authorization token for your user. --wandb flag required.", default=None, type=str)
    parser.add_argument('--wandb_id', help="Resume the sweep that has this ID. --wandb flag required.", default=None, type=str)
    parser.add_argument('--runs', help="Number of sweeping runs. Not used during grid search. --wandb flag required.", default=None, type=int)
    args = parser.parse_args()

    # Integration with W&B
    if args.wandb is not None:
        if args.wandb_key is None:
            raise ValueError("W&B was flagged for use but no token was provided. Please insert your token with the command line flag '--wandb_key'.")
        os.environ['WANDB_API_KEY'] = args.wandb_key
    else:
        raise ValueError("W&B needed to configure the sweeps for hyperparameter optimizatinon.")

    with open(os.path.join(os.getcwd().split("cv-gender-age-recognition")[0]+"cv-gender-age-recognition/", "code/models/ours/config.yaml"), 'r') as configfile:
        hyperparameters=yaml.safe_load(configfile)

    sweep_id = wandb.sweep(hyperparameters, entity=args.wandb, project="gar_itw")
    if args.runs is not None:
        wandb.agent(sweep_id, sweep_wrapper, count=args.runs)
    else:
        wandb.agent(sweep_id, sweep_wrapper)
