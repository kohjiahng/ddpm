"""modules for UNet
"""
import math
from configparser import ConfigParser
from torch import Tensor
import torch
from torch import nn

config = ConfigParser()
config.read('config.ini')
IMG_RES = config.getint('params', 'IMG_RES')
class ResBlock(nn.Module):
    """Residual block
    This takes in a 4D tensor along with a time embedding and outputs a 4D tensor of the same shape
    """
    def __init__(self, filters: int, dropout: bool = False) -> None:
        """

        Args:
            filters (int): number of in/out channels
            dropout (bool, optional): Whether to apply dropout after instancenorm. Defaults to False.
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(filters, filters, bias=False, kernel_size=3, padding='same'),
            nn.InstanceNorm2d(filters, affine=True),
            *([nn.Dropout(0.5)] if dropout else []),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(filters, filters, bias=False, kernel_size=3, padding='same'),
            nn.InstanceNorm2d(filters, affine=True),
            *([nn.Dropout(0.5)] if dropout else []),
            nn.SiLU()
        )
        self.embed_layer = nn.Sequential(
            nn.LazyLinear(filters),
            nn.SiLU()
        )
    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        """forward pass

        Args:
            x (Tensor): 4D tensor of batched images
            temb (Tensor): 2D tensor of batched time embeddings

        Returns:
            Tensor: Output of block with the same shape as input x
        """
        tproj = self.embed_layer(temb)
        y = self.conv1(x)
        y += tproj[:,:,None,None]
        return self.conv2(y) + x
class MultiheadSelfAttention(nn.MultiheadAttention):
    """Wrapper over nn.MultiheadAttention for self-attention
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    def forward(self, x: Tensor) -> Tensor:
        # pylint: disable=arguments-differ
        return super().forward(x,x,x,need_weights=False)[0]
class AttentionBlock(nn.Module):
    """Pixel-wise self attention followed by instancenorm and silu
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attn = MultiheadSelfAttention(embed_dim=channels, num_heads=1, batch_first=True)

        self.norm = nn.Sequential(
            nn.InstanceNorm2d(channels,affine=True),
            nn.SiLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        """forward pass

        Args:
            x (Tensor): 4D tensor of batched inputs

        Returns:
            Tensor: Output of block with same shape as input x
        """
        height, width = x.shape[2:]
        flattened_x = torch.flatten(x, start_dim=2) # N x C x HW
        flattened_x = torch.swapaxes(flattened_x, 1, 2) # N x HW x C
        flattened_out = self.attn(flattened_x) 
        flattened_out = torch.swapaxes(flattened_out, 1, 2)
        out = flattened_out.unflatten(-1, (height,width))
        out = self.norm(out)
        assert x.shape == out.shape
        return x + out

class ResAttentionBlock(nn.Module):
    """ResidualBlock with attention
    """
    def __init__(self, in_channels: int, out_channels: int | None = None,
                 attn: bool = False) -> None:
        """

        Args:
            in_channels (int): number of input channels
            out_channels (int | None, optional): number of output channels. Defaults to None.
            attn (bool, optional): whether to include attention block. Defaults to False.
        """
        super().__init__()
        self.resblock = ResBlock(in_channels)
        self.attn = attn
        self.out_channels = out_channels
        if attn:
            self.attnblock = AttentionBlock(in_channels)
        if out_channels is not None:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        """forward pass

        Args:
            x (Tensor): 4D tensor of batched inputs
            temb (Tensor): 2D tensor of batched time embeddings

        Returns:
            Tensor: 4D tensor of output of block with out_channels channels
        """
        x = self.resblock(x, temb)
        if self.attn:
            x = self.attnblock(x)
        if self.out_channels is not None:
            x = self.conv(x)

        return x


class TimeEncoding(nn.Module):
    """TimeEncoding
    """
    def __init__(self, time_dim: int, max_len: int = 1000) -> None:
        """

        Args:
            time_dim (int): dimension to embed into
            max_len (int, optional): maximum time. Defaults to 1000.
        """
        super().__init__()
        assert time_dim % 2 == 0
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, time_dim, 2) * (-math.log(10000.0) / time_dim))
        pe = torch.zeros(max_len, time_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, t: Tensor) -> Tensor:
        """Generates embeddings for time t

        Args:
            t (Tensor): 1D tensor of times

        Returns:
            Tensor: N x time_dim tensor of embeddings
        """        """"""
        return self.pe[t-1,:]
 