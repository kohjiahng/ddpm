"""UNet with residual blocks and attention
"""
from configparser import ConfigParser
import torch
from torch import nn, Tensor, IntTensor
import numpy as np
from modules import TimeEncoding, ResAttentionBlock

config = ConfigParser()
config.read('config.ini')

IMG_RES = config.getint('params','IMG_RES')
def block_downsample(in_channels: int, out_channels: int,
                     n_blocks: int = 2, attn: bool = False) -> nn.Sequential:
    """Downsample block (downsamples by 2)

    Args:
        in_channels (int): input channels
        out_channels (int): output channels
        n_blocks (int, optional): Number of ResAttention blocks. Defaults to 2.
        attn (bool, optional): whether to include attention in blocks. Defaults to False.

    Returns:
        nn.Sequential: Series of processing layers, then downsample
    """
    enc_kwargs = {
        'kernel_size': (4,4),
        'stride': 2,
        'padding': 1,
    }
    resattnblocks = [
        ResAttentionBlock(in_channels, attn=attn) for _ in range(n_blocks)
    ]
    return nn.Sequential(
        *resattnblocks,
        nn.Conv2d(in_channels, out_channels,**enc_kwargs)
    )

def block_upsample(in_channels: int, out_channels: int,
                   n_blocks: int = 2, attn: bool = False) -> nn.Sequential:
    """Upsample block (upsamples by 2)

    Args:
        in_channels (int): input channels
        out_channels (int): output channels
        n_blocks (int, optional): Number of ResAttention blocks. Defaults to 2.
        attn (bool, optional): whether to include attention in blocks. Defaults to False.

    Returns:
        nn.Sequential: Series of processing layers, then downsample
    """
    dec_kwargs = {
        'kernel_size': (4,4),
        'stride': 2,
        'padding': 1,
    }
    resattnblocks = [
        ResAttentionBlock(out_channels*2, out_channels, attn=attn) for _ in range(n_blocks)
    ]
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels,**dec_kwargs),
        *resattnblocks
    )


class UNet(nn.Module):
    """UNet with residuals and attention
    """
    def __init__(self, hid_channels: int = 128, time_dim: int = 256, max_t: int = 1000) -> None:
        """

        Args:
            hid_channels (int, optional): Base number of channels in hidden layers. Defaults to 128.
            time_dim (int, optional): Dimension of time encoding. Defaults to 256.
            max_t (int, optional): Maximum time input. Defaults to 1000.
        """
        super().__init__()

        self.time_encoder = TimeEncoding(time_dim, max_t)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3,hid_channels,kernel_size=(1,1)),
            nn.SiLU()
        ) # 64*256*256
        self.downsample = nn.Sequential(
            *block_downsample(hid_channels, hid_channels), # 128*128*128
            *block_downsample(hid_channels, hid_channels), # 256*64*64
            *block_downsample(hid_channels, hid_channels), # 512*32*32
            *block_downsample(hid_channels, hid_channels, attn=True), # 1024*16*16
        )

        # self.downsample = [
        #     block_downsample_pair(hid_channels, hid_channels*2), # 128*128*128
        #     block_downsample_pair(hid_channels*2, hid_channels*4), # 256*64*64
        #     block_downsample_pair(hid_channels*4, hid_channels*8), # 512*32*32
        #     block_downsample_pair(hid_channels*8, hid_channels*16, attn=True), # 1024*16*16
        #     block_downsample_pair(hid_channels*16, hid_channels*32, attn=True), # 2048*8*8
        # ]
        self.middle = nn.ModuleList([
            ResAttentionBlock(hid_channels, attn=True),
            ResAttentionBlock(hid_channels, attn=True),
        ])

        self.upsample = nn.Sequential(
            *block_upsample(hid_channels*2, hid_channels, attn=True), # 512*32*32
            *block_upsample(hid_channels*2, hid_channels), # 256*64*64
            *block_upsample(hid_channels*2, hid_channels), # 128*128*128
            *block_upsample(hid_channels*2, hid_channels), # 64*256*256
        )
        self.feature_block = nn.Conv2d(hid_channels, 3, kernel_size=(1,1))

    def forward(self, x: Tensor, t: IntTensor | np.ndarray[int]) -> Tensor:
        """Forward pass of UNet

        Args:
            x (Tensor): 4D tensor of batched inputs
            t (IntTensor | np.ndarray[int]): 1D object of times

        Returns:
            Tensor: _description_
        """
        temb = self.time_encoder(t)
        x = self.initial_conv(x)

        skips = []
        for block in self.downsample:
            if isinstance(block, ResAttentionBlock):
                x = block(x, temb)
            else:
                x = block(x)
            # skips.append(x)

        for block in self.middle:
            x = block(x, temb)
        skips = reversed(skips)
        # for block, skip in zip(self.upsample, skips):
        for block in self.upsample:
            x = torch.cat([x, x], dim = 1)
            if isinstance(block, ResAttentionBlock):
                x = block(x, temb)
            else:
                x = block(x)

        return self.feature_block(x)


if __name__ == '__main__':
    net = UNet().cuda()
    inp = torch.zeros((1,3,IMG_RES,IMG_RES)).cuda()
    out = net(inp, torch.zeros(1,dtype=int)+3)
    size = sum([param.numel() for param in net.parameters() if param.requires_grad])
    print(f"UNet output shape: {out.shape}")
    print(f"UNet size: {size/1e6:.3}M parameters")
    print(f"Memory usage for one image: {torch.cuda.max_memory_allocated()/1e9:.3}GB")
    print(net)
