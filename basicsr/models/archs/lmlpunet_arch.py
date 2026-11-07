import torch
from torch import nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import Upsample, ACmix
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


# def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
#
#
# def shift(dim):
#     x_shift = [torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
#     x_cat = torch.cat(x_shift, 1)
#     x_cat = torch.narrow(x_cat, 2, self.pad, H)
#     x_cat = torch.narrow(x_cat, 3, self.pad, W)
#     return x_cat

class convBlock(nn.Module):
    def __init__(self, dim, out_dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, out_dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Conv2d(dim, out_dim, 1)  # if drop_path > 0. else nn.Identity()

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

        x = input + self.drop_path(x)
        return x


class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=7):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    #     def shift(x, dim):
    #         x = F.pad(x, "constant", 0)
    #         x = torch.chunk(x, shift_size, 1)
    #         x = [ torch.roll(x_c, shift, dim) for x_s, shift in zip(x, range(-pad, pad+1))]
    #         x = torch.cat(x, 1)
    #         return x[:, :, pad:-pad, pad:-pad]

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)

        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)

        x = self.fc1(x_shift_r)

        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x


class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


def uppad(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2))
    return x1


class hr_dense(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=8, num_grow_ch=8):
        super(hr_dense, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)

        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)

        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1,
                               1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1,
                               1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.bn = nn.BatchNorm2d(num_grow_ch)

    def forward(self, x):
        x1 = self.lrelu(self.bn(self.conv1(x)))
        x2 = self.lrelu(self.bn(self.conv2(torch.cat((x, x1), 1))))
        x3 = self.lrelu(self.bn(self.conv3(torch.cat((x, x1, x2), 1))))
        x4 = self.lrelu(self.bn(self.conv4(torch.cat((x, x1, x2, x3), 1))))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class LMLPUNet(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self, num_classes=3, input_channels=3, deep_supervision=False, img_size=224, patch_size=16,
                 in_chans=3,
                 embed_dims=[128, 160, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        self.first = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder1 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.first_2 = ACmix(3, 16, 5, stride=1, padding=2)
        self.encoder1_2 = ACmix(16, 16, 5, stride=1, padding=2)
        self.encoder2_2 = ACmix(16, 32, 5, stride=1, padding=2)
        self.encoder3_2 = ACmix(32, 128, 5, stride=1, padding=2)
        self.encoder4_2 = ACmix(128, 160, 5, stride=1, padding=2)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                              in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                              in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        # self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        # self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        # self.decoder3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        # self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        # self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.upsample1 = Upsample(scale=2, num_feat=160)
        self.upsample2 = Upsample(scale=2, num_feat=128)
        self.upsample3 = Upsample(scale=2, num_feat=32)
        self.upsample4 = Upsample(scale=2, num_feat=16)
        self.upsample5 = Upsample(scale=2, num_feat=16)

        self.decoder1 = ACmix(256, 160, 3, stride=1, padding=1)
        self.decoder2 = ACmix(160, 128, 3, stride=1, padding=1)
        self.decoder3 = ACmix(128, 32, 3, stride=1, padding=1)
        self.decoder4 = ACmix(32, 16, 3, stride=1, padding=1)
        self.decoder5 = ACmix(16, 16, 3, stride=1, padding=1)

        self.local_final = nn.Conv2d(16, 16, 3, 1, 1)

        # upsample
        self.conv_up1 = nn.Conv2d(16, 16, 3, 1, 1)

        self.conv_hr = nn.Conv2d(16, 16, 3, 1, 1)
        # self.conv_hr = hr_dense()

        self.conv_last = nn.Conv2d(16, num_classes, 3, 1, 1)
        # self.conv_out = nn.Conv2d(num_classes, num_classes, 3, 1, 1)
        self.conv_out = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, 3, 1, 1),
            nn.Conv2d(num_classes, num_classes, 3, 1, 1),
            nn.Conv2d(num_classes, num_classes, 3, 1, 1),
            nn.Conv2d(num_classes, num_classes, 3, 1, 1), )

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.lrelu = nn.GELU()
        self.upsample_in = Upsample(scale=4, num_feat=3)
        self.upsample = Upsample(scale=4, num_feat=16)
        self.sigmoid = nn.Sigmoid()
        # self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        residual_global = x
        x = self.first(x)
        x_2 = self.first_2(residual_global)
        residual = x
        residual_2 = x_2

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1 CNN 3*3+max_pooling
        # out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))  # /2
        out = self.lrelu(F.max_pool2d((self.encoder1(x)), 2, 2))  # /2
        out_2 = self.lrelu(F.max_pool2d((self.encoder1_2(x_2)), 2, 2))  # /2
        t1 = out
        t1_2 = out_2
        ### Stage 2  CNN 3*3+max_pooling
        # out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))  # /4
        out = self.lrelu(F.max_pool2d((self.encoder2(out)), 2, 2))  # /4
        out_2 = self.lrelu(F.max_pool2d((self.encoder2_2(out_2)), 2, 2))  # /4
        t2 = out
        t2_2 = out_2
        ### Stage 3  CNN 3*3+max_pooling
        # out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))  # /8
        out = self.lrelu(F.max_pool2d((self.encoder3(out)), 2, 2))  # /8
        out_2 = self.lrelu(F.max_pool2d((self.encoder3_2(out_2)), 2, 2))  # /8
        t3 = out
        t3_2 = out_2

        ### Tokenized MLP Stage
        ### Stage 4

        out_2 = self.lrelu(F.max_pool2d((self.encoder4_2(out_2)), 2, 2))  # /8
        t4_2 = out_2

        out, H, W = self.patch_embed3(out)  # 生成Token 3*3 输出通道即是Token数目   # /16
        for i, blk in enumerate(self.block1):  # shift MLP
            out = blk(out, H, W)
        # out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out, H, W = self.patch_embed4(out)  # /32
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        # out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4

        # out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2), mode='bilinear', align_corners=True))  # /16
        out = self.lrelu(F.interpolate((self.decoder1(out)), scale_factor=(2, 2), mode='bilinear',
                                       align_corners=True))  # /16
        # out = self.lrelu((self.upsample1((self.decoder1(out)))))  # /16
        out = uppad(out, t4)
        # out = self.sigmoid(out) * out
        # t4 = self.sigmoid(t4) * t4
        out = torch.add(out, t4)
        # out = uppad(out, t4_2)
        # t4_2 = self.sigmoid(t4_2) * t4_2
        # out = torch.add(out, t4_2)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3

        # out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear', align_corners=True))  # /8
        out = self.lrelu(F.interpolate((self.decoder2(out)), scale_factor=(2, 2), mode='bilinear',
                                       align_corners=True))  # /8
        # out = self.lrelu(self.upsample2((self.decoder2(out))))  # /8
        out = uppad(out, t3)
        # out = self.sigmoid(out) * out
        # t3 = self.sigmoid(t3) * t3
        out = torch.add(out, t3)
        # out = uppad(out, t3_2)
        # t3_2 = self.sigmoid(t3_2) * t3_2
        # out = torch.add(out, t3_2)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        # out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear', align_corners=True))  # /4
        out = self.lrelu(F.interpolate((self.decoder3(out)), scale_factor=(2, 2), mode='bilinear',
                                       align_corners=True))  # /4
        # out = self.lrelu(self.upsample3((self.decoder3(out))))  # /4
        out = uppad(out, t2)
        # out = self.sigmoid(out) * out
        # t2 = self.sigmoid(t2) * t2
        out = torch.add(out, t2)
        # out = uppad(out, t2_2)
        # t2_2 = self.sigmoid(t2_2) * t2_2
        # out = torch.add(out, t2_2)
        # out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear', align_corners=True))  # /2
        out = self.lrelu(F.interpolate((self.decoder4(out)), scale_factor=(2, 2), mode='bilinear',
                                       align_corners=True))  # /2
        # out = self.lrelu(self.upsample4((self.decoder4(out))))  # /2
        out = uppad(out, t1)
        # out = self.sigmoid(out) * out
        # t1 = self.sigmoid(t1) * t1
        out = torch.add(out, t1)
        # out = uppad(out, t1_2)
        # t1_2 = self.sigmoid(t1_2) * t1_2
        # out = torch.add(out, t1_2)

        # out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))  # /HW
        out = self.lrelu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))  # /HW
        # out = self.lrelu(self.upsample5(self.decoder5(out)))  # /HW
        out = self.local_final(out)
        out = uppad(out, residual)
        out = out + residual  # local residual
        # out = uppad(out, residual_2)
        # out = out + residual_2  # local residual

        # global out  coarse to fine
        feat = self.lrelu(self.conv_up1(self.upsample(out)))
        out = self.conv_last(feat)  # self.lrelu(self.conv_last(feat))
        feat_in = self.upsample_in(residual_global)
        out = out + feat_in  # global residual
        out = self.conv_out(out)
        return out

# if __name__ == '__main__':
#     ift = torch.randn((32, 3, 65, 65))
#     sft = MLPUNet()
#     out = sft(ift)
#     print(out.size(), ift.size())
#     print(torch.sum(out - ift))
