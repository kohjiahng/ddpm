"""Utility functions
"""

import torch
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

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
