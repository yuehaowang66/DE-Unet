import torch
import torch.nn as nn
import torch.nn.functional as F
from model.EMA import EMA

# -----------------------------
# Basic blocks
# -----------------------------

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class FUConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.ema = EMA(channels=out_channels)

        self.dilated_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=4, dilation=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=4, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.ema_dilated = EMA(channels=out_channels)   

        self.fusion = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x_main = self.double_conv(x)
        x_main = self.ema(x_main)

        x_dilated = self.dilated_conv(x)
        x_dilated = self.ema_dilated(x_dilated)

        x_cat = torch.cat([x_main, x_dilated], dim=1)  
        out = self.fusion(x_cat)
        return out


class Down(nn.Module):
    """Downscaling with maxpool then FUConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            FUConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# -----------------------------
# Integrated EMSAA 
# -----------------------------

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups if groups > 0 else num_channels
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError(f'activation layer [{act}] is not found')
    return layer


class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.up_dwc(x)
        groups = max(1, self.in_channels)
        x = channel_shuffle(x, groups)
        x = self.pwc(x)
        return x


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MultiScaleFuse(nn.Module):
    def __init__(self, channels, num_branches=3, reduction=4):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels * num_branches, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, num_branches, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.num_branches = num_branches

    def forward(self, features):
        # features: [ [B,C,H,W], ... ] len=branches
        x = torch.cat(features, dim=1)    # [B, C*branches, H, W]
        w = self.global_pool(x)           # [B, C*branches, 1, 1]
        w = self.fc1(w)                   # [B, C//r, 1, 1]
        w = self.relu(w)
        w = self.fc2(w)                   # [B, num_branches, 1, 1]
        w = self.softmax(w)               # [B, num_branches, 1, 1]
        out = 0
        for i, f in enumerate(features):
            out = out + f * w[:, i:i+1, ...]
        return out


class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super().__init__()
        dim = max(1, int(out_channels // factor))
        groups = max(1, dim // 4)
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        self.conv_3x3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=groups)  
        self.conv_5x5 = nn.Conv2d(dim, dim, 5, 1, 2, groups=groups)
        self.conv_7x7 = nn.Conv2d(dim, dim, 7, 1, 3, groups=groups)
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim)
        self.fuse = MultiScaleFuse(dim)
        self.up = nn.Conv2d(dim, out_channels, 1, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2, x4):
        x_fused = torch.cat([x1, x2, x4], dim=1)
        x_fused = self.down(x_fused)
        x_fused_c = x_fused * self.channel_attention(x_fused)
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = self.fuse([x_3x3, x_5x5, x_7x7])
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)
        out = self.up(self.relu(self.bn(x_fused_s + x_fused_c + x_fused))) 
        return out


class EMSAA(nn.Module):

    def __init__(self, dec_in, enc2, enc3, enc4, out_channels, activation='relu'):
        super(EMSAA, self).__init__()
        self.eucb = EUCB(dec_in, out_channels, activation=activation)
        self.em_fuse = FusionConv(enc2 + enc3 + enc4, out_channels)
        self.post_fuse = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x_dec, enc_feats):

        dec_up = self.eucb(x_dec)              # [B, out, H', W']
        target_size = dec_up.shape[2:]

        f2, f3, f4 = enc_feats
        f2 = F.interpolate(f2, size=target_size, mode='bilinear', align_corners=False)
        f3 = F.interpolate(f3, size=target_size, mode='bilinear', align_corners=False)
        f4 = F.interpolate(f4, size=target_size, mode='bilinear', align_corners=False)
        enc_fused = self.em_fuse(f2, f3, f4)   # [B, out, H', W']

        out = torch.cat([dec_up, enc_fused], dim=1)
        out = self.post_fuse(out)              # [B, out, H', W']
        return out


class Up(nn.Module):

    def __init__(self, in1, enc2, enc3, enc4, out_channels, bilinear=True):
        super().__init__()
        self.emsaa = EMSAA(in1, enc2, enc3, enc4, out_channels)
        self.bilinear = bilinear

    def forward(self, x1, enc_feats):
        return self.emsaa(x1, enc_feats)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
