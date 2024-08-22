"""Diffusion model with pytorch lightning
"""
import logging
from typing import Callable
from configparser import ConfigParser
import numpy as np
from torch import Tensor, IntTensor
from torch import nn
import torch
import pytorch_lightning as L
import matplotlib.pyplot as plt
import wandb
from utils import plot_images, Mean

config = ConfigParser()
config.read('config.ini')

IMG_RES = config.getint('params', 'IMG_RES')

class DiffusionModel(L.LightningModule):
    """Class to implement diffusion

    Args:
        create_net (Callable[[], nn.Module]): function to create NN to model noise
        T (int, optional): Total timesteps. Defaults to 1000.
        optimizer (torch.optim.Optimizer, optional): Optimizer class. Defaults to torch.optim.Adam.
        opt_config (dict, optional): optimizer config to be passed into optimizer. Defaults to {}.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, create_net: Callable[[], nn.Module], T: int = 1000,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 opt_config: dict | None = None) -> None:
        super().__init__()
        self.max_timesteps = T
        self.net = create_net()
        # UNet(hid_channels, max_t = T)
        self.optimizer = optimizer
        self.opt_config = (opt_config if opt_config is not None else {})
        self.mse_loss = nn.MSELoss()

        self.example_input_array = Tensor(1, 3, IMG_RES, IMG_RES) # for lightning test passes

        # ---------------------------------- Logging --------------------------------- #
        self.log_config = {
            'n_unconditional': 3,
            'freq_unconditional': 100,
            't_reconstructed': 100,
            'freq_loss_by_time': 100
        }
        self.train_loss_metric = Mean()
        self.val_loss_metric = Mean()

        freq = self.log_config['freq_loss_by_time']
        self.val_loss_by_time_metrics = {
            t: Mean() for t in np.arange(self.max_timesteps+1, step=freq, dtype=int)
        }


        # Resolve all lazy modules
        self.net(
            torch.zeros((1,3,IMG_RES,IMG_RES), device=self.device),
            torch.ones(1, dtype=int, device='cpu') # array of indexes must be on cpu
        )
        wandb.watch(self.net)

        # ----------------------------- Diffusion params ----------------------------- #
        self.beta = np.linspace(1e-4, 0.02, self.max_timesteps) # linear schedule
        self.alpha = 1-self.beta
        self.alpha_bar = np.cumprod(self.alpha)

        self.sigma = np.sqrt( # backward variance
            (1 - self.alpha_bar[:-1]) * self.beta[1:] / (1 - self.alpha_bar[1:])
        )
        self.sigma = np.array([0, *self.sigma])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers (lightning hook)

        Returns:
            torch.optim.Optimizer: Optimizer for training
        """
        return self.optimizer(self.net.parameters(), **self.opt_config)

    def forward_sample(self, x: Tensor,
                       t: int | IntTensor | np.ndarray[int]) -> tuple[Tensor, Tensor]:
        """Sample x_{t+1} given x_t

        Args:
            x (Tensor): 4D tensor of batched images
            t (int/IntTensor/array): 1D object of times, or a single timestep for all images

        Returns:
            Tensor: Sampled x_{t+1}, noise parameter from N(0,I)
        """
        eps = torch.normal(0, 1, x.shape, device=self.device)
        x_coef = np.sqrt(self.alpha_bar[t-1]).reshape(-1,1,1,1)
        eps_coef = np.sqrt(1-self.alpha_bar[t-1]).reshape(-1,1,1,1)
        x_coef = torch.tensor(x_coef, dtype=torch.float, device=self.device)
        eps_coef = torch.tensor(eps_coef, dtype=torch.float, device=self.device)

        noised_x = x_coef * x + eps_coef * eps

        return noised_x, eps
    def backward_sample(self, x: Tensor, t: int, return_eps: bool = False) -> Tensor:
        """Sample x{t-1} given x_t

        Args:
            x (Tensor): 4D tensor of batched images
            t (int): Timestep of images
            return_eps (bool, optional): Whether to return raw model output. Defaults to False

        Returns:
            Tensor | tuple[Tensor, Tensor]: Sampled x_{t-1} optionally with raw model output
        """
        eps = self.net(x,torch.full((x.shape[0],), t)) 
        eps_coef = (1-self.alpha[t-1]) / np.sqrt(1-self.alpha_bar[t-1])

        mu = (x - eps_coef * eps) / np.sqrt(self.alpha[t-1])

        if t > 0:
            prev_x = torch.normal(mu, self.sigma[t-1])
        else:
            prev_x = mu

        if return_eps:
            return prev_x, eps
        return prev_x
    def decode_noise(self, x: Tensor,
                     log_timesteps: list[Tensor] | None = None,
                     cur_timestep: int | None = None) -> list[Tensor] | Tensor:
        """full backward process from timestep T

        Args:
            x (Tensor): 4D tensor of batched images
            cur_timestep (int | None, optional): Timestep of x or self.max_timesteps if unspecified Defaults to None.
            log_timesteps (list, optional): Timesteps (apart from 0) to return. Defaults to None.

        Returns:
            (list[Tensor]/Tensor): List of x at log_timesteps and 0 (Reverse timestep order). Single tensor if no log_timesteps is given
        """

        if cur_timestep is None:
            cur_timestep = self.max_timesteps
        if log_timesteps is None:
            log_timesteps = []

        assert x.shape[1:] == (3,IMG_RES,IMG_RES)

        out = []
        for t in range(cur_timestep,0,-1):
            if t in log_timesteps:
                out.append(x.cpu())
            x = self.backward_sample(x, t)
        out.append(x.cpu())

        if len(out) == 1:
            return out[0]
        else:
            return out
    def get_losses_by_time(self, x: Tensor, times: list[int]) -> list[float]:
        """Get L2 losses over sampled time

        Args:
            x (Tensor): 4D tensor of batched images
            times (list[int]): list of times to get loss

        Returns:
            list[float]: losses for each time in times
        """

        losses_by_time = []
        for t in times:
            noised_batch, eps = self.forward_sample(x, t)
            pred = self.net(noised_batch, torch.full((noised_batch.shape[0],), t))
            loss = self.mse_loss(pred, eps)
            losses_by_time.append(loss.cpu())
        return losses_by_time
    def generate_images(self, n: int,
                        timesteps: list[int] | np.ndarray[int] | None = None) -> list[Tensor] | Tensor:
        """generate images, optionally with intermediate values

        Args:
            n (int): number of images to generate
            timesteps (list[int] | np.ndarray[int] | None): timesteps to return

        Returns:
            list[Tensor] | Tensor: list of 4D tensors of generated images,
                                   or a single 4D tensor if timesteps is not specified
        """
        if timesteps is None:
            timesteps = []

        x = torch.normal(0,1,(n, 3, IMG_RES, IMG_RES),device=self.device)
        generated_xs = self.decode_noise(x, timesteps)
        return generated_xs
    def progressive_generation(self, n: int,
                               timesteps: list[int] | np.ndarray[int]) -> list[Tensor]:
        """generate x0 progressively by estimating x0 at each step

        Args:
            n (int): number of images to generate
            timesteps (list[int] | np.ndarray[int]): timesteps to return

        Returns:
            list[Tensor]: list of 4D tensors of generated images
        """

        if len(timesteps) == 0:
            logging.warning("DffusionModel.progressive_generation called with empty timesteps, use generate_images instead")
        generated = []
        x = torch.normal(0,1,(n, 3, IMG_RES, IMG_RES),device=self.device)
        for t in range(self.max_timesteps,0,-1):
            prev_x, eps = self.backward_sample(x, t, return_eps=True)
            if t in timesteps:
                x_hat = (x - np.sqrt(1-self.alpha_bar[t-1])*eps) / np.sqrt(self.alpha_bar[t-1])
                generated.append(x_hat.cpu())
            x = prev_x
        generated.append(x.cpu())
        return generated
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward diffusion process (pytorch hook)

        Args:
            x (Tensor): 4D tensor of batched images

        Returns:
            (Tensor, Tensor): Predicted noise and real noise parameters
        """

        # pylint: disable=arguments-differ

        t = np.random.randint(1,self.max_timesteps+1,size=x.shape[0])

        noised_x, eps = self.forward_sample(x, t)

        # predict the noise
        return self.net(noised_x, t), eps

    def training_step(self, x: Tensor) -> Tensor:
        """Training step (lightning hook)

        Args:
            x (Tensor): 4D tensor of batched images

        Returns:
            Tensor: L2 loss
        """

        # pylint: disable=arguments-differ

        out, eps = self.forward(x)
        loss = self.mse_loss(out, eps)
        self.train_loss_metric(loss.cpu().item())
        return loss
    def on_train_epoch_end(self):
        """Log metrics after epoch (lightning hook)
        """
        loss = self.train_loss_metric.result()
        self.log('train/loss', loss)
        self.log('train/RMSE', np.sqrt(loss))
        self.train_loss_metric.clear()

    def on_validation_start(self) -> None:
        """log unconditional generated images (pytorch hook)
        """

        n = self.log_config['n_unconditional']
        freq = self.log_config['freq_unconditional']

        # Log generated images
        timesteps = np.arange(self.max_timesteps+1, step=freq, dtype=int)
        generated_images = self.generate_images(n, timesteps)
        gen_fig = plot_images(generated_images)
        wandb.log({'generated': gen_fig})

        # Log progressive generation
        prog_gen = self.progressive_generation(n, timesteps)
        prog_gen_fig = plot_images(prog_gen)
        wandb.log({'val/progressive': prog_gen_fig})
    def on_first_validation_batch_start(self, batch: Tensor) -> None:
        """Called on the first validation batch

        Args:
            batch (Tensor): 4D tensor of batched images
        """
        # Log reconstructed images (noise partially then backward)

        t_recon = self.log_config['t_reconstructed'] #100 # step to noise until
        noised_batch, _ = self.forward_sample(batch, t_recon)
        recon_batch = self.decode_noise(noised_batch,cur_timestep=t_recon)
        recon_fig = plot_images(
            [batch.cpu(), noised_batch.cpu(), recon_batch.cpu()]
        )
        wandb.log({'val/reconstructed': recon_fig})
    def on_validation_batch_start(self, batch: Tensor, batch_idx: int) -> None:
        """log reconstructed images for first validation batch (pytorch hook)

        Args:
            batch (Tensor): 4D tensor of batched images
            batch_idx (int): index of batch
        """
        # pylint: disable=arguments-differ
        if batch_idx == 0:
            self.on_first_validation_batch_start(batch)
    def validation_step(self, batch: Tensor) -> None:
        """validation step (lightning hook)

        Args:
            batch (Tensor): 4D tensor of batched images
        """
        # pylint: disable=arguments-differ

        # get validation loss
        out, eps = self.forward(batch)
        loss = self.mse_loss(out, eps)
        self.val_loss_metric(loss.cpu().item())
       
        # Update losses by time metric
        timesteps = self.val_loss_by_time_metrics.keys()
        batch_losses_by_time = self.get_losses_by_time(batch, timesteps)
        for t, loss in zip(timesteps, batch_losses_by_time):
            self.val_loss_by_time_metrics[t](loss)
    def on_validation_epoch_end(self) -> None:
        """log accumulated validation loss and losses by time
        """
        loss = self.val_loss_metric.result()
        self.log('val/loss', loss)
        self.log('val/rmse', np.sqrt(loss))

        val_losses_by_time = [
            metric.result() for metric in self.val_loss_by_time_metrics.values()
        ]
        timesteps = self.val_loss_by_time_metrics.keys()

        fig = plt.figure()
        plt.plot(timesteps, val_losses_by_time)
        wandb.log({'val/losses_by_time': fig})

        self.val_loss_metric.clear()
        for metric in self.val_loss_by_time_metrics.values():
            metric.clear()

if __name__ == '__main__':
    from dataset import DataModule
    from net2 import UNet
    torch.random.manual_seed(0)
    model = DiffusionModel(UNet)
    dm = DataModule('./data/PetImages/Cat', 1, 1)
    dm.setup('fit')
    dl = dm.train_dataloader()
    img = next(iter(dl))
    noised_img, _ = model.forward_sample(img, np.array([400]))
    plt.imshow((noised_img[0].permute((1,2,0))+1)/2)
    plt.show()
