"""Utility functions
"""

import statistics
import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytorch_lightning as L

# ------------ infer_type convenience function for config logging ------------ #
def infer_type(val: str) -> int | float | str:
    """convenience function for config logging

    Args:
        val (str): _description_

    Returns:
        int | float | str: inferred typecast of val
    """
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val

def channel_first(x: Tensor) -> Tensor:
    """Helper function to move channel first for images

    Args:
        x (Tensor): 4D tensor of batched channel-last images

    Returns:
        Tensor: x with channels first
    """
    return torch.permute(x, (0,3,1,2))
def channel_last(x: Tensor) -> Tensor:
    """Helper function to move channel last for images

    Args:
        x (Tensor): 4D tensor of batched channel-first images

    Returns:
        Tensor: x with channels last
    """
    return torch.permute(x, (0,2,3,1))

def plot_images(images: list[Tensor]) -> Figure:
    """Plot a batch of images over time

    Args:
        images (list[Tensor]): List of 4D tensors of batched images (channel-first). Each image must be of equal shape.

    Returns:
        Figure: plotted images with x axis being batch dimension and y axis time
    """
    for image_batch in images:
        assert image_batch.shape == images[0].shape

    n = len(images)
    m = len(images[0])
    fig, ax = plt.subplots(n,m,figsize=(m*8,n*8))

    for t_idx, image_batch in enumerate(images):
        image_batch = torch.clamp(image_batch, -1, 1)

        image_batch = channel_last(image_batch)
        for idx, img in enumerate(image_batch):
            ax[t_idx, idx].imshow((img+1)/2)
            ax[t_idx, idx].axis('off')
        fig.subplots_adjust(wspace=0,hspace=0.1)
    return fig

class Mean:
    """Helper class to accumulate mean
    """
    def __init__(self) -> None:
        self.sum = 0
        self.cnt = 0
    def __call__(self, x: float) -> None:
        self.sum += x
        self.cnt += 1
    def result(self) -> float:
        """Return current mean

        Returns:
            float: mean
        """
        return self.sum / self.cnt
    def clear(self) -> None:
        """Reset mean
        """
        self.sum = 0
        self.cnt = 0

class MeanSeries:
    """Helper class to accumulate a series of means
    """

    def __init__(self, max_index: int) -> None:
        self.max_index = max_index
        self.sum = np.zeros(self.max_index + 1)
        self.cnt = np.zeros(self.max_index + 1, dtype=int)

    def __call__(self, index: int, val: float) -> None:
        """Update an entry to the series

        Args:
            index (int): index of mean
            val (float): value to update

        Raises:
            ValueError: If invalid index
        """
        if index < 0 or index > self.max_index:
            raise ValueError("index passed to MeanSeries out of range")

        self.sum[index] += val
        self.cnt[index] += 1

    def result(self) -> tuple[np.ndarray[int], np.ndarray[float]]:
        """Get list of means, filtered by existence

        Returns:
            tuple[list[int], list[float]]: indices with at least one entry and their means
        """
        means = self.sum / self.cnt
        filtered_means = means[self.cnt > 0]
        filtered_index, _ = np.nonzero(self.cnt)
        return filtered_index, filtered_means

class EpochTimer(L.Callback):
    """A callback that logs the epoch execution time for train and val."""
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def on_train_epoch_start(self, trainer: L.Trainer, *args, **kwargs):
        self.start.record()

    def on_train_epoch_end(self, trainer: L.Trainer, *args, **kwargs):
        # Exclude the first iteration to let the model warm up
        if trainer.global_step > 1:
            self.end.record()
            torch.cuda.synchronize()
            time = self.start.elapsed_time(self.end) / 1000
            trainer.logger.log('train/epoch_time', time)
    def on_validation_epoch_start(self, trainer: L.Trainer, *args, **kwargs):
        self.start.record()

    def on_validation_epoch_end(self, trainer: L.Trainer, *args, **kwargs):
        # Exclude the first iteration to let the model warm up
        if trainer.global_step > 1:
            self.end.record()
            torch.cuda.synchronize()

            time = self.start.elapsed_time(self.end) / 1000
            trainer.logger.log('val/epoch_time', time)