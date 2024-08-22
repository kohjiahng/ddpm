"""UNet Architecture coppied from CycleGAN
"""
from configparser import ConfigParser
import torch
from torch import nn
from modules import TimeEncoding
config = ConfigParser()
config.read('config.ini')

IMG_RES = config.getint('params','IMG_RES')

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

class UNet(nn.Module):
    def __init__(self, hid_channels=64, time_dim = 256, max_t = 1000):
        super().__init__()

        self.time_encoder = TimeEncoding(time_dim, max_t)

        enc_kwargs = {
            'kernel_size': (4,4),
            'stride': 2,
            'padding': 1,
        }

        self.enc = nn.Sequential(
            ConvInstanceNormRelu(hid_channels,norm=False,**enc_kwargs),
            ConvInstanceNormRelu(hid_channels*2,**enc_kwargs),
            ConvInstanceNormRelu(hid_channels*4,**enc_kwargs),
            ConvInstanceNormRelu(hid_channels*8,**enc_kwargs),
            ConvInstanceNormRelu(hid_channels*8,**enc_kwargs),
            ConvInstanceNormRelu(hid_channels*8,**enc_kwargs),
            ConvInstanceNormRelu(hid_channels*8,**enc_kwargs),
            ConvInstanceNormRelu(hid_channels*8,norm=False,**enc_kwargs),
        )
        dec_kwargs = {
            'kernel_size': (4,4),
            'stride': 2,
            'padding': 1
        }
        self.dec = nn.Sequential(
            ConvTransposeInstanceNormRelu(hid_channels*8,**dec_kwargs,dropout=True),
            ConvTransposeInstanceNormRelu(hid_channels*8,**dec_kwargs,dropout=True),
            ConvTransposeInstanceNormRelu(hid_channels*8,**dec_kwargs,dropout=True),
            ConvTransposeInstanceNormRelu(hid_channels*8,**dec_kwargs),
            ConvTransposeInstanceNormRelu(hid_channels*4,**dec_kwargs),
            ConvTransposeInstanceNormRelu(hid_channels*2,**dec_kwargs),
            ConvTransposeInstanceNormRelu(hid_channels,**dec_kwargs),
        )
        self.feature_block = nn.Sequential(
            nn.ConvTranspose2d(hid_channels*2, 3, kernel_size=(4,4), stride=2,padding=1),
            # nn.Tanh()
        )

    def forward(self, x, t):
        t_enc = self.time_encoder(t)

        skips = []
        for down in self.enc:
            x = down(x, t_enc)
            # print(x.shape)
            skips.append(x)
        skips = reversed(skips[:-1])
        for up, skip in zip(self.dec, skips):
            x = up(x, t_enc)
            x = torch.cat([x, skip], dim = 1)
        return self.feature_block(x)

    
class MLP(nn.Module):
    def __init__(self, hid_channels=64, time_dim = 256, max_t = 1000):
        super().__init__()

        self.time_encoder = TimeEncoding(time_dim, max_t)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Sequential(
            nn.LazyLinear(100),
            nn.ReLU()
        )
        self.lin2 = nn.Sequential(
            nn.LazyLinear(100),
            nn.ReLU(),
        )
        self.lin3 = nn.LazyLinear(3*28*28)
    def forward(self, x, t):
        t_enc = self.time_encoder(t)
        x = self.flatten(x)
        x = torch.concat([x, t_enc], dim=1)
        x = self.lin1(x)

        x = torch.concat([x, t_enc], dim=1)
        x = self.lin2(x)

        return self.lin3(x).reshape((-1, 3,28,28))
if __name__ == '__main__':
    gen = UNet().cuda()
    inp = torch.zeros((2, 3, IMG_RES, IMG_RES)).cuda()
    out = gen(inp, torch.tensor([1,2]))
    print(f"UNet output shape: {out.shape}")