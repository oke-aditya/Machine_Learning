import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim

class LitModel(pl.LightningModule):
    def __init__(self,):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)
        return result

        # return {'loss': loss, 'log': {'train_loss': loss}}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, prog_bar=True)
        return result
        # return {'loss': loss, 'log': {'val_loss': loss}}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {'loss': loss, 'log': {'test_loss': loss}}

    # def validation_epoch_end(self, validation_step_outputs):
    #     all_val_losses = validation_step_outputs.val_loss
    #     all_predictions = validation_step_outputs.predictions

class mnist_datamodule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
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
    model = LitModel()
    # dataset = MNIST(os.getcwd(), download=True, transform=T.ToTensor())
    
    # train_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    # val_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # From mnist data module
    dm = mnist_datamodule(batch_size=32)

    trainer = pl.Trainer(max_epochs=1, gpus=None)
    # trainer.fit(model, train_loader, val_loader)
    trainer.fit(model, dm)

