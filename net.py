from configparser import ConfigParser
from modules import ConvInstanceNormRelu, ConvTransposeInstanceNormRelu, TimeEncoding
from torch import nn
import torch
config = ConfigParser()
config.read('config.ini')

IMG_RES = config.getint('params','IMG_RES')

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
            skips.append(x)
        skips = reversed(skips[:-1])
        for up, skip in zip(self.dec, skips):
            x = up(x, t_enc)
            x = torch.cat([x, skip], dim = 1)
        return self.feature_block(x)

    
if __name__ == '__main__':
    gen = UNet()
    inp = torch.zeros((1, 3, IMG_RES, IMG_RES))
    out = gen(inp)
    print(f"UNet output shape: {out.shape}")