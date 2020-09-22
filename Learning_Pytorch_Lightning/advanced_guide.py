import os
import pytorch_lightning as pl
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(x.shape(0), -1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        result = pl.TrainResult(loss)
        return result
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        result = pl.EvalResult(loss)
        return result
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        result = pl.EvalResult(loss)
        return result
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3)

class mnist_datamoudule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        MNIST(os.getcwd(), train=True, download=True)
        MNIST(os.getcwd(), train=False, download=True)
    
    # Call for every GPU 
    def setup(self, stage):
        transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.13, ), (0.308))
        ])

        if stage == "fit":
            mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transforms)
            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        
        if stage == "test":
            self.mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms)
    
    def train_dataloader(self):
        mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=False, )
        return mnist_train

    def val_dataloader(self):
        mnist_val = DataLoader(self.mnist_val, batch_size=self.batch_size, shuffle=False, )
        return mnist_val
    
    def test_dataloader(self):
        mnist_test = DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False, )
        return mnist_test

if __name__ == "__main__":
    dm = mnist_datamoudule()
    model = LitModel()
    trainer = pl.Trainer(gpus=None, max_epochs=1)
    trainer.fit(model, dm)
    
