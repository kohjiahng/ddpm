import torch
import math
from torch import nn
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')
IMG_RES = config.getint('params', 'IMG_RES')

class ConvInstanceNormRelu(nn.Module):
    def __init__(self, filters, leaky=True, norm=True,**kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.LazyConv2d(filters, bias=not norm,**kwargs),
            *([nn.InstanceNorm2d(filters, affine=True)] if norm else []),
            nn.LeakyReLU(0.2) if leaky else nn.ReLU()
        )
        self.embed_layer = nn.Sequential(
            nn.LazyLinear(filters),
            nn.ReLU()
        )
    def forward(self, X, t):
        return self.block(X) + self.embed_layer(t).unsqueeze(-1).unsqueeze(-1)

class ConvTransposeInstanceNormRelu(nn.Module):
    def __init__(self, filters, dropout=False, **kwargs): 
        super().__init__()
        self.block = nn.Sequential(
            nn.LazyConvTranspose2d(filters,bias=False, **kwargs),
            nn.InstanceNorm2d(filters, affine=True),
            *([nn.Dropout(0.5)] if dropout else []),
            nn.ReLU()
        )
        self.embed_layer = nn.Sequential(
            nn.LazyLinear(filters),
            nn.ReLU()
        )
    def forward(self, X, t):
        return self.block(X) + self.embed_layer(t).unsqueeze(-1).unsqueeze(-1)

class TimeEncoding(nn.Module):
    def __init__(self, time_dim: int, max_len: int = 1000):
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
            t: int

            Output: Tensor(time_dim, )
        """
        return self.pe[t-1,:]