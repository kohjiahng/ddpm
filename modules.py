import torch
import math
from torch import nn
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')
IMG_RES = config.getint('params', 'IMG_RES')
class ResBlock(nn.Module):
    def __init__(self, filters, dropout=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(filters, filters, bias=False, kernel_size=3, padding='same'),
            nn.InstanceNorm2d(filters, affine=True),
            *([nn.Dropout(0.5)] if dropout else []),
            nn.SiLU()
        )
        self.embed_layer = nn.Sequential(
            nn.LazyLinear(filters),
            nn.SiLU()
        )
    def forward(self, x, temb):
        tproj = self.embed_layer(temb)
        x += tproj[:,:,None,None]
        return self.block(x) + x
class MultiheadSelfAttention(nn.MultiheadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, x):
        return super().forward(x,x,x,need_weights=False)[0]
class AttentionBlock(nn.Module):
    def __init__(self, channels, dim):
        super().__init__()
        self.block = nn.Sequential(
            MultiheadSelfAttention(embed_dim=channels, num_heads=channels, batch_first=True),
            nn.SiLU()
        )

    def forward(self, x):
        N, C, H, W = x.shape
        flattened_x = torch.flatten(x, start_dim=2) # N x C x HW
        flattened_x = torch.swapaxes(flattened_x, 1, 2) # N x HW x C
        flattened_out = self.block(flattened_x) 
        flattened_out = torch.swapaxes(flattened_out, 1, 2)
        out = flattened_out.unflatten(-1, (H,W))
        assert x.shape == out.shape
        return x + out
        
class ResAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, attn=False, attn_dim=16):
        super().__init__()
        self.resblock = ResBlock(in_channels)
        self.attn = attn
        self.out_channels = out_channels
        if attn:
            self.attnblock = AttentionBlock(in_channels, attn_dim)
        if out_channels is not None:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        
    def forward(self, x, temb, skip = None):
        x = self.resblock(x, temb)
        if self.attn:
            x = self.attnblock(x)
        if self.out_channels is not None:
            x = self.conv(x)

        return x


class TimeEncoding(nn.Module):
    def __init__(self, time_dim: torch.Tensor, max_len: int = 1000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, time_dim, 2) * (-math.log(10000.0) / time_dim))
        pe = torch.zeros(max_len, time_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, t):
        """
            Generates embeddings for time t
            t: 1d Tensor

            Output: Tensor(time_dim, )
        """
        return self.pe[t-1,:]