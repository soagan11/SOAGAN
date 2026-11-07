import torch
from torch import nn as nn

from basicsr.models.archs.arch_util import Upsample, make_layer, RCB, SpatialAttention, CCALayer, ESA


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            ContextBlock(num_feat),
            # nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.Conv2d(num_feat, num_feat, 1, padding=0),
            nn.ReLU(inplace=True),
            # nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Conv2d(num_feat, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class rbconv3(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(rbconv3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, 3, 1, 1),
        )
        self.act = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        out = self.conv(x) + x
        out = self.act(out)
        return out


class convBlock(nn.Module):
    def __init__(self, dim, out_dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        # self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 2 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.pwconv2 = nn.Linear(2 * dim, out_dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Conv2d(dim, out_dim, 3, 1, 1)  # if drop_path > 0. else nn.Identity()
        self.ca = ChannelAttention(dim)
        self.conv = rbconv3(dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.ca(self.act(self.conv(x)))
        x = x + self.drop_path(input)
        return x


class ContextBlock(nn.Module):

    def __init__(self, n_feat, bias=True):
        super(ContextBlock, self).__init__()

        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        )

    def modeling(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        # return x + context
        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.modeling(x)

        # [N, C, 1, 1]
        return self.channel_add_conv(context)
        # channel_add_term = self.channel_add_conv(context)
        # x = x + channel_add_term
        # return x


class RCB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=True, groups=1):
        super(RCB, self).__init__()

        act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential(
            # nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=n_feat),
            # act,
            convBlock(n_feat, n_feat),
            # rbconv3(n_feat, n_feat),
            # rbconv3(n_feat, n_feat),
            # nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            # nn.Conv2d(n_feat, n_feat, 1),
            # nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=n_feat),
            # act,
            # nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            # nn.Conv2d(n_feat, n_feat, 3, 1, 1),
        )

        # self.act = act
        # self.act = rbconv3(n_feat, n_feat)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(n_feat, n_feat, 3, 1, 1),
        #     act,
        #     nn.Conv2d(n_feat, n_feat, 3, 1, 1),
        #     act,
        #     nn.Conv2d(n_feat, n_feat, 3, 1, 1),
        # )
        # self.gcnet = ContextBlock(n_feat, bias=bias)
        # self.ca = ChannelAttention(n_feat)
        # self.ca = CCALayer(n_feat)
        # self.sa = ESA(n_feat)
        # self.sa = SpatialAttention()
        self.a = nn.Parameter(torch.tensor(1.))
        # self.out = convBlock(n_feat, n_feat)
        # self.out_last = nn.Conv2d(n_feat, n_feat, 3, 1, 1)

    def forward(self, x):
        res = self.body(x)
        # res = res + x
        # res = self.act(self.gcnet(res))
        # res = self.sa(res) + self.ca(res)
        # res = self.out_last(res)

        return res * self.a + x
        # return res


class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, squeeze_factor=16, res_scale=1):
        super(RCAB, self).__init__()
        self.res_scale = res_scale

        self.rcab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            # rbconv3(num_feat, num_feat),
            RCB(num_feat),
            # SpatialAttention(),
            # ESA(num_feat),
            # ChannelAttention(num_feat, squeeze_factor)
            # nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            # nn.Conv2d(num_feat, num_feat, 3, 1, 1),
        )
        # self.sa = nn.Sequential(
        #     # nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ReLU(True),
        #     # nn.Conv2d(num_feat, num_feat, 3, 1, 1),
        #     rbconv3(num_feat, num_feat),
        #     SpatialAttention(),
        #     nn.Conv2d(num_feat, num_feat, 1, 1),
        #     # ESA(num_feat),
        #     # ChannelAttention(num_feat, squeeze_factor)
        # )
        # self.conv = nn.Sequential(
        #     nn.Conv2d(num_feat, num_feat, 3, 1, 1),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv2d(num_feat, num_feat, 3, 1, 1),
        # )

    def forward(self, x):
        res = self.rcab(x) * self.res_scale
        # res = self.sa(res)
        # res = self.conv(res)
        return res + x


class ResidualGroup(nn.Module):
    """Residual Group of RCAB.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_block (int): Block number in the body network.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, num_block, squeeze_factor=16, res_scale=1):
        super(ResidualGroup, self).__init__()

        self.residual_group = make_layer(
            RCAB,
            num_block,
            num_feat=num_feat,
            squeeze_factor=squeeze_factor,
            res_scale=res_scale)
        self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        res = self.conv(self.residual_group(x))
        return res + x


class RCAN(nn.Module):
    """Residual Channel Attention Networks.

    Paper: Image Super-Resolution Using Very Deep Residual Channel Attention
        Networks
    Ref git repo: https://github.com/yulunzhang/RCAN.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_group (int): Number of ResidualGroup. Default: 10.
        num_block (int): Number of RCAB in ResidualGroup. Default: 16.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_group=10,
                 num_block=16,
                 squeeze_factor=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(RCAN, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(
            ResidualGroup,
            num_group,
            num_feat=num_feat,
            num_block=num_block,
            squeeze_factor=squeeze_factor,
            res_scale=res_scale)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean

        return x
