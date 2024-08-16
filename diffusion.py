import numpy as np
from configparser import ConfigParser
from torch import nn
import torch
import pytorch_lightning as L
import logging
import wandb
from net import UNet
from torch.distributions.multivariate_normal import MultivariateNormal
from utils import plot_images
import matplotlib.pyplot as plt
config = ConfigParser()
config.read('config.ini')

IMG_RES = config.getint('params', 'IMG_RES')

class DiffusionModel(L.LightningModule):
    def __init__(self, T=1000, optimizer=torch.optim.Adam, opt_config={}):
        super().__init__()
        self.T = T
        self.net = UNet()
        self.optimizer = optimizer
        self.beta = np.linspace(1e-4, 0.02, T)
        self.alpha = 1-self.beta
        self.alpha_bar = np.cumprod(self.alpha)

        self.opt_config = opt_config
        self.mse_loss = nn.MSELoss()
        self.example_input_array = torch.Tensor(1, 3, 256, 256)

        self.train_losses = []
        self.sigma = np.sqrt(
            (1 - self.alpha_bar[:-1]) * self.beta[1:] / (1 - self.alpha_bar[1:])
        )
        
    def forward(self, x):
        eps = torch.normal(0, 1, x.shape, device=self.device)
        t = np.random.randint(1,self.T)

        noised_x = np.sqrt(self.alpha_bar[t-1]) * x + np.sqrt(1-self.alpha_bar[t-1]) * eps

        # predict the noise
        return self.net(noised_x, t), eps
    
    def init_weights(self):
        pass

    def setup(self, stage):
        if stage == "fit":
            self.init_weights()
    def configure_optimizers(self):
        return self.optimizer(self.net.parameters(), **self.opt_config)
    
    def training_step(self, batch, batch_idx):
        out, eps = self(batch)
        loss = self.mse_loss(out, eps)
        self.train_losses.append(loss.cpu().item())
        return loss
    def on_train_epoch_end(self):
        loss = sum(self.train_losses) / len(self.train_losses)
        self.log('train/loss', loss)
        self.train_losses = []
    def validation_step(self,num_sample):
        breakpoints = np.linspace(self.T, 0, 3)

        history = []
        x = torch.normal(0,1,(num_sample, 3, IMG_RES, IMG_RES),device=self.device)
        for t in range(self.T,0,-1):
            if t in breakpoints:
                history.append(x.cpu())

            out = self.net(x, t)
            mu = 1/np.sqrt(self.alpha[t-1]) * (x - (1-self.alpha[t-1]) * out /np.sqrt(1-self.alpha_bar[t-1]))
            if t > 0:
                x = torch.normal(mu, self.sigma[t-1])
            else:
                x = mu
        history.append(x.cpu())
        fig = plot_images(history, breakpoints)
        wandb.log({'images': fig})
        return x