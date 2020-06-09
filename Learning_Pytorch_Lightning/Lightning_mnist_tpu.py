
# To install the TPU stuff

# !curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
# !python pytorch-xla-env-setup.py --version $VERSION

# ! pip -q install pytorch-lightning

import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

class mnist_cnn(pl.lightning_module):

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 10)
    
    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size[0], -1)))
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        train_loss = F.cross_entropy(y_hat, y)
        return {"train_loss" : train_loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        return {"val_loss" : val_loss}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        return {"test_loss" : test_loss}
    
    def configure_optimizers(self, x):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
    
    def prepare_data(self):
        self.mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        self.mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    def train_dataloader(self):
        loader = DataLoader(self.mnist_train, batch_size=32, num_workers=4)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.mnist_test, batch_size=32, num_workers=4)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.mnist_test, batch_size=32, num_workers=4)
        return loader

