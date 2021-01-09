# The contents of this file are adapted from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import BFBatchNorm2d
from models import register_model


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
            self,
            in_channels,
            out_channels,
            mid_channels=None,
            bias=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, bias=bias),
            nn.BatchNorm2d(mid_channels) if bias else BFBatchNorm2d.BFBatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, bias=bias),
            nn.BatchNorm2d(out_channels) if bias else BFBatchNorm2d.BFBatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, bias=bias)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, bias=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of
        # channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True)
            self.conv = DoubleConv(
                in_channels,
                out_channels,
                in_channels // 2,
                bias=bias)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2,
                bias=bias)
            self.conv = DoubleConv(in_channels, out_channels, bias=bias)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias)

    def forward(self, x):
        return self.conv(x)


""" Full assembly of the parts to form the complete network """


@register_model("unet2")
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, scales = 6, base_channels = 16, residual = False, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear
        self.scales = scales
        self.residual = residual
        assert self.scales in [2, 4, 6]

        factor = 2 if bilinear else 1
        
        self.inc = DoubleConv(self.n_channels, base_channels, bias=bias)
        self.down1 = Down(base_channels, base_channels, bias=bias)
        self.down2 = Down(base_channels, base_channels, bias=bias)
        if self.scales >= 4:
            self.down3 = Down(base_channels, base_channels*2, bias=bias)
            self.down4 = Down(base_channels*2, base_channels*4 if self.scales == 6 else base_channels*2, #to make sure dimensions of up3 match 
							  bias=bias)
			
            if self.scales == 6:
                self.down5 = Down(base_channels*4, base_channels*8, bias=bias)
                self.down6 = Down(base_channels*8, base_channels*16 // factor, bias=bias)
        
                self.up1 = Up(base_channels*16, base_channels*8 // factor, bilinear, bias=bias)
                self.up2 = Up(base_channels*8, base_channels*4 // factor, bilinear, bias=bias)
                
            self.up3 = Up(base_channels*4, base_channels*2 // factor, bilinear, bias=bias)
            self.up4 = Up(base_channels*2, base_channels, bilinear, bias=bias)
            
        self.up5 = Up(base_channels*2, base_channels, bilinear, bias=bias)
        self.up6 = Up(base_channels*2, base_channels, bilinear, bias=bias)
        
        self.outc = OutConv(base_channels, self.n_classes, bias=bias)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--in-channels",
            type=int,
            default=1,
            help="number of input channels")
        parser.add_argument(
            "--out-channels",
            type=int,
            default=1,
            help="number of output channels")
        parser.add_argument(
            "--bias",
            action="store_true",
            help="use bias")
        parser.add_argument(
            "--residual",
            action="store_true",
            help="use residual connection")
        parser.add_argument(
            "--scales",
            type=int,
            default=6,
            help="number of unet scales")
        parser.add_argument(
            "--base-channels",
            type=int,
            default=16,
            help="number of channels in each convolutional layer")

    @classmethod
    def build_model(cls, args):
        return cls(args.in_channels, args.out_channels, bias = args.bias, 
                   scales = args.scales, base_channels = args.base_channels, 
                   residual = args.residual if hasattr(args, 'residual') else False)

    def forward(self, y):
        x1 = self.inc(y)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
		
        if self.scales >= 4:
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            
            if self.scales == 6:
                x6 = self.down5(x5)
                x7 = self.down6(x6)
                
                x = self.up1(x7, x6)
                x = self.up2(x, x5)
            else:
                x = x5
                
            x = self.up3(x, x4)
            x = self.up4(x, x3)
        else:
            x = x3
            
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        logits = self.outc(x)
        
        if self.residual:
            return y - logits
        else:
            return logits
