# Importing the required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type

# --- Base Double Convolution ---
class BaseDoubleConv(nn.Module):
    """Standard Convolution Block"""
    def __init__(self, in_channels, out_channels, mid_channels=None, num_groups=32):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class BaseDown(nn.Module):
    """
    Downscaling with L2 pooling then double conv.
    """
    def __init__(self, in_channels, out_channels, conv_op: Type[nn.Module] = BaseDoubleConv):
        super().__init__()
        self.l2pool_conv = nn.Sequential(
            nn.LPPool2d(norm_type=2, kernel_size=2),
            conv_op(in_channels, out_channels)
        )

    def forward(self, x):
        return self.l2pool_conv(x)

class BaseOutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BaseOutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# --- STANDARD U-NET ---
class StandardUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = BaseDoubleConv(in_channels, out_channels, num_groups=32)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Standard_UNet(nn.Module):
    def __init__(self, n_channels=9, n_out=2):
        super(Standard_UNet, self).__init__()
        self.inc = BaseDoubleConv(n_channels, 64, num_groups=32)
        self.down1 = BaseDown(64, 128)
        self.down2 = BaseDown(128, 256)
        self.down3 = BaseDown(256, 512)
        self.up1 = StandardUp(512, 256)
        self.up2 = StandardUp(256, 128)
        self.up3 = StandardUp(128, 64)
        self.outc = BaseOutConv(64, n_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)

# --- RESIDUAL U-NET ---
class ResidualDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, out_channels)
        ) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))

class ResidualUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ResidualDoubleConv(in_channels, out_channels, num_groups=32)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Residual_UNet(nn.Module):
    def __init__(self, n_channels=9, n_out=2):
        super(Residual_UNet, self).__init__()
        self.inc = ResidualDoubleConv(n_channels, 64, num_groups=32)
        self.down1 = BaseDown(64, 128, conv_op=ResidualDoubleConv)
        self.down2 = BaseDown(128, 256, conv_op=ResidualDoubleConv)
        self.down3 = BaseDown(256, 512, conv_op=ResidualDoubleConv)
        self.up1 = ResidualUp(512, 256)
        self.up2 = ResidualUp(256, 128)
        self.up3 = ResidualUp(128, 64)
        self.outc = nn.Conv2d(64, n_out, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)

# --- ATTENTION U-NET ---
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, num_groups=32):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups, F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups, F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=False)
        return x * self.psi(self.relu(g1 + x1))

class AttentionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.attn = AttentionGate(F_g=in_channels // 2, F_l=in_channels // 2, F_int=in_channels // 4)
        self.conv = BaseDoubleConv(in_channels, out_channels, num_groups=32)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2_attn = self.attn(g=x1, x=x2)
        diffY, diffX = x2_attn.size()[2] - x1.size()[2], x2_attn.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2_attn, x1], dim=1)
        return self.conv(x)

class Attention_UNet(nn.Module):
    def __init__(self, n_channels=9, n_out=2):
        super(Attention_UNet, self).__init__()
        self.inc = BaseDoubleConv(n_channels, 64, num_groups=32)
        self.down1 = BaseDown(64, 128)
        self.down2 = BaseDown(128, 256)
        self.down3 = BaseDown(256, 512)
        self.up1 = AttentionUp(512, 256)
        self.up2 = AttentionUp(256, 128)
        self.up3 = AttentionUp(128, 64)
        self.outc = BaseOutConv(64, n_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)

# --- CBAM U-NET ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))

class CBAMDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, num_groups=32):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True)
        )
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.conv_block(x)
        x = self.ca(x) * x
        return self.sa(x) * x

class CBAM_UNet(nn.Module):
    def __init__(self, n_channels=9, n_out=2):
        super(CBAM_UNet, self).__init__()
        self.inc = CBAMDoubleConv(n_channels, 64, num_groups=32)
        self.down1 = BaseDown(64, 128, conv_op=CBAMDoubleConv)
        self.down2 = BaseDown(128, 256, conv_op=CBAMDoubleConv)
        self.down3 = BaseDown(256, 512, conv_op=CBAMDoubleConv)
        self.up1 = StandardUp(512, 256)
        self.up2 = StandardUp(256, 128)
        self.up3 = StandardUp(128, 64)
        self.outc = BaseOutConv(64, n_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)