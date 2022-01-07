# Imports
import wandb
import sys
import copy
import numpy as np

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms


class feature_extractor(nn.Module):
    def __init__(self, resnet_version, coupling_size):
        super(feature_extractor,self).__init__()
        self.__load_resnet__(resnet_version)
        self.resnet.fc = nn.Linear(512, coupling_size)

    def forward(self, x):
        return self.resnet(x)
    
    def __load_resnet__(self, version):
        if type(version) is str:
            version = int(version)
        elif type(version) not in [int, str]:
            sys.exit("ResNet version argument must be an integer or string in [18, 34, 50, 101, 152].")
        if version == 18:
            self.resnet = models.resnet18()
        elif version == 34:
            self.resnet = models.resnet34()
        elif version == 50:
            self.resnet = models.resnet50()
        elif version == 101:
            self.resnet = models.resnet101()
        elif version == 152:
            self.resnet = models.resnet152()
        else:
            sys.exit(f"""ResNet{version} is not available.\n
                         Choose one of [18, 34, 50, 101, 152].""")


class classifier(nn.Module):
    def __init__(self, coupling_size, dropout):
        super(classifier,self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(coupling_size, 128),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(32, 8),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.MLP(x)
        return self.sigmoid(x).squeeze(1)


class GenRec(nn.Module):
    def __init__(self, resnet_version=18, coupling_size=512, dropout=0):
        super(GenRec,self).__init__()
        torch.manual_seed(1234)
        self.best_acc = 0
        self.FE = feature_extractor(resnet_version, coupling_size)
        self.MLP = classifier(coupling_size, dropout)

    def forward(self, x):
        x = self.FE(x)
        x = self.MLP(x)
        return x

    def num_params(self, trainable=True):
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable)
    
    def train_model(self, dloaders, optimizer, hparams, wb_run=None, prints=False):
        loss_fn = nn.BCELoss()
        wandb.watch(self, loss_fn, log="all", log_freq=50) if wb_run else None
        va_losses, va_accs, tr_losses, tr_accs = [], [], [], []
        best_model = copy.deepcopy(self.state_dict())
        for epoch in range(hparams.num_epochs):
            print(f"Epoch {epoch+1}/{hparams.num_epochs}") if prints else None
            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.train()
                else:
                    self.eval()
                accum_loss, accum_acc = 0, 0
                for x, y in dloaders[phase]:
                    x, y = x.to(hparams.device), y.to(hparams.device)
                    optimizer.zero_grad()
                    out = self.forward(x)
                    loss = loss_fn(out, y.float())
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    # Training stats
                    accum_loss += loss.item() * x.size(0)
                    preds = torch.round(out).detach()
                    accum_acc += torch.sum(preds == y).item()
                n = len(dloaders[phase].dataset)
                epoch_loss, epoch_acc = accum_loss/n, accum_acc/n
                if prints:
                    print('[{}] Loss: {:.4f} | Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                if wb_run:
                    wandb.log({f"{phase}_loss":epoch_loss, f"{phase}_acc":epoch_acc}, step=epoch)
                if phase == "valid":
                    va_losses.append(epoch_loss); va_accs.append(epoch_acc)
                    if self.best_acc < epoch_acc:
                        self.best_acc = epoch_acc
                        best_model = copy.deepcopy(self.state_dict())
                else:
                    tr_losses.append(epoch_loss); tr_accs.append(epoch_acc)
            wandb.log({"best_valid_acc": self.best_acc}) if wb_run else None
        self.load_state_dict(best_model)
        return {"valid": {"loss": va_losses, "acc": va_accs}, "train": {"loss": tr_losses, "acc": tr_accs}}

# Variables to be imported
torchdevice = "cuda" if torch.cuda.is_available() else "cpu"

from utils import root

model = GenRec()
model.load_state_dict(torch.load(f"{root}checkpoints/pretrained_model_checkpoint.pth", map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
