import torch
import torch.nn as nn
from timm.models.layers import DropPath
from torch.nn import Module, ModuleList, \
    Sequential, \
    Linear, \
    LayerNorm, \
    Conv2d, \
    BatchNorm2d, \
    ReLU, \
    GELU, \
    Identity
from basicsr.models.archs.arch_util import make_layer, Upsample, ChannelAttention, SKFF, ACmix, convBlock, ASPP, SKConv
from einops import rearrange
import numbers
import torch.nn.functional as F


class ConvTokenizer(nn.Module):
    def __init__(self, embedding_dim=64):
        super(ConvTokenizer, self).__init__()
        self.first = nn.Conv2d(3, embedding_dim, 3, 1, 1)
        self.block = nn.Sequential(
            nn.Conv2d(embedding_dim,
                      embedding_dim,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      ),
            nn.PReLU(),
            nn.Conv2d(embedding_dim,
                      embedding_dim,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      ),
            nn.PReLU(),
            nn.Conv2d(embedding_dim,
                      embedding_dim,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      ),
            nn.PReLU(),
        )

    def forward(self, x):
        residual = self.first(x)
        return self.block(residual) + residual


class ConvStage(Module):
    def __init__(self,
                 num_blocks=2,
                 embedding_dim_in=64,
                 hidden_dim=128,
                 embedding_dim_out=128):
        super(ConvStage, self).__init__()
        self.conv_blocks = ModuleList()
        for i in range(num_blocks):
            block = Sequential(
                Conv2d(embedding_dim_in,
                       hidden_dim,
                       kernel_size=(3, 3),
                       stride=(1, 1),
                       padding=(1, 1),
                       ),
                nn.PReLU(),
                Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                # BatchNorm2d(hidden_dim),
                nn.PReLU(),
                Conv2d(hidden_dim, embedding_dim_in, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.PReLU(),
            )
            self.conv_blocks.append(block)
        self.downsample = Conv2d(embedding_dim_in,
                                 embedding_dim_out,
                                 kernel_size=(3, 3),
                                 stride=(1, 1),
                                 padding=(1, 1))

    def forward(self, x):
        residual = x
        for block in self.conv_blocks:
            x = block(x)
        # return self.downsample(x) + residual
        return x + residual


class Mlp(Module):
    def __init__(self,
                 embedding_dim_in,
                 hidden_dim=None,
                 embedding_dim_out=None,
                 activation=GELU):
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim_in
        embedding_dim_out = embedding_dim_out or embedding_dim_in
        self.fc1 = Linear(embedding_dim_in, hidden_dim)
        self.act = activation()
        self.fc2 = Linear(hidden_dim, embedding_dim_out)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='BiasFree'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        # self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1)

        # self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
        #                         groups=hidden_features * 2)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2)  # ,
        # groups=hidden_features)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1)  # ,
        # groups=hidden_features)
        self.dwconv3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1)  # , groups=hidden_features)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1)
        # self.sigmoid = nn.Sigmoid()
        self.sigmoid = F.gelu

    def forward(self, x):
        residual = x
        x = self.project_in(x)  # NCHW
        # x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x3 = self.dwconv3(x)
        x = self.sigmoid(x3) * x2
        x = self.sigmoid(x) * x1
        x = self.project_out(x)
        return x + residual


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # q = torch.nn.functional.normalize(q, dim=-1)
        # k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out + x


class ConvMLP(Module):
    def __init__(self, embedding_dim):
        self.norm1 = LayerNorm(embedding_dim)
        self.channel_mlp1 = Mlp(embedding_dim_in=embedding_dim, hidden_dim=embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)
        self.connect = Conv2d(embedding_dim,
                              embedding_dim,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=(1, 1),
                              groups=embedding_dim,
                              bias=False)
        self.connect_norm = LayerNorm(embedding_dim)
        self.channel_mlp2 = Mlp(embedding_dim_in=embedding_dim, hidden_dim=dim_feedforward)

    def forward(self, src):
        h, w = src.shape[-2:]
        to_4d(self.body(to_3d(src)), h, w)

        src = src + to_4d(self.norm1(self.channel_mlp1(to_3d(src))), h, w)
        src = self.connect(src)
        src = src + to_4d(self.norm2(self.channel_mlp2(to_3d(src))), h, w)
        return src


class ConvMLPStage(Module):
    def __init__(self,
                 embedding_dim,
                 dim_feedforward=2048,
                 stochastic_depth_rate=0.1):
        super(ConvMLPStage, self).__init__()
        # self.norm1 = LayerNorm(embedding_dim)
        # self.channel_mlp1 = Mlp(embedding_dim_in=embedding_dim, hidden_dim=dim_feedforward)
        # self.norm2 = LayerNorm(embedding_dim)
        # self.connect = Conv2d(embedding_dim,
        #                       embedding_dim,
        #                       kernel_size=(3, 3),
        #                       stride=(1, 1),
        #                       padding=(1, 1),
        #                       groups=embedding_dim,
        #                       bias=False)
        # self.connect_norm = LayerNorm(embedding_dim)
        # self.channel_mlp2 = Mlp(embedding_dim_in=embedding_dim, hidden_dim=dim_feedforward)
        # self.drop_path = DropPath(stochastic_depth_rate) if stochastic_depth_rate > 0 else Identity()

        # self-atttention modul
        self.body = nn.Sequential(
            # LayerNorm(embedding_dim),
            # FeedForward(embedding_dim),
            nn.Conv2d(embedding_dim, embedding_dim, 3, 1, 1),
            SKConv(embedding_dim, embedding_dim),
            nn.Conv2d(embedding_dim, embedding_dim, 1, 1),
            # LayerNorm(embedding_dim),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.GELU(),
            nn.Conv2d(embedding_dim, embedding_dim, 3, 1, 1),
            SKConv(embedding_dim, embedding_dim),
            nn.Conv2d(embedding_dim, embedding_dim, 1, 1),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.GELU(),
            # nn.Conv2d(embedding_dim, embedding_dim, 1, 1),
            # nn.Conv2d(embedding_dim, embedding_dim, 3, 1, 1),
            # nn.Conv2d(embedding_dim, embedding_dim, 1, 1),
            # SKConv(embedding_dim, embedding_dim),
            # nn.GELU(),
            # SKConv(embedding_dim, embedding_dim),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # convBlock(embedding_dim, embedding_dim),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # FeedForward(embedding_dim),
            # LayerNorm(embedding_dim),
            # nn.Conv2d(embedding_dim, embedding_dim, 3, 1, 1, 1),
            # nn.Conv2d(embedding_dim, embedding_dim, 3, 1, 1, 1, embedding_dim),
            # Attention(embedding_dim),
            # ACmix(embedding_dim, embedding_dim),
            # LayerNorm(embedding_dim),
            # FeedForward(embedding_dim),
            # convBlock(embedding_dim, embedding_dim),
            # convBlock(embedding_dim, embedding_dim),
            # ASPP(embedding_dim, embedding_dim),
            # ASPP(embedding_dim, embedding_dim),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # ACmix(embedding_dim, embedding_dim),
            # nn.Conv2d(embedding_dim, embedding_dim, 3, 1, 1, 1),
            # nn.Conv2d(embedding_dim, embedding_dim, 3, 1, 1, 1),
        )
        self.skff = SKFF(embedding_dim, 2)

    def forward(self, src):
        # src = src + self.drop_path(self.norm1(self.channel_mlp1(src)))
        # src = self.connect(self.connect_norm(src).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # src = src + self.drop_path(self.norm2(self.channel_mlp2(src)))
        residual = src
        # src = self.body(src)
        # src = self.body(src)
        src = self.body(src) + residual
        # src = self.body(src) + residual
        # src = self.body(src)
        # src = self.skff([src, residual])
        return src


class ConvDownsample(Module):
    def __init__(self, embedding_dim_in, embedding_dim_out):
        super().__init__()
        self.downsample = Conv2d(embedding_dim_in, embedding_dim_out, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1))

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)  # NHWC->NCHW
        x = self.downsample(x)
        # return x.permute(0, 2, 3, 1)
        return x


class BasicStage(Module):
    def __init__(self,
                 num_blocks,
                 embedding_dims,
                 mlp_ratio=1,
                 stochastic_depth_rate=0.1,
                 downsample=True):
        super(BasicStage, self).__init__()
        self.blocks = ModuleList()
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_blocks)]
        for i in range(num_blocks):
            block = ConvMLPStage(embedding_dim=embedding_dims[0],
                                 dim_feedforward=int(embedding_dims[0] * mlp_ratio),
                                 stochastic_depth_rate=dpr[i],
                                 )
            self.blocks.append(block)

        self.downsample_mlp = ConvDownsample(embedding_dims[0], embedding_dims[1]) if downsample else Identity()
        # self.informationpool = nn.Conv2d((num_blocks + 1) * embedding_dims[0], embedding_dims[0], 3, 1, 1)
        self.skff = SKFF(embedding_dims[0], num_blocks + 1)
        # self.skff_last = SKFF(embedding_dims[0], 3)
        self.prelu = nn.PReLU()

    def forward(self, x):  # input NCHW
        residual = x
        # out = residual
        out = []
        for blk in self.blocks:
            x = blk(x)
            out.append(x)
            # x = x + residual
            # out = torch.cat((out, x), dim=1)
            # x = x + residual
        # x = self.downsample_mlp(x)
        out.append(residual)
        x = self.prelu(self.skff(out))  # self.informationpool(out)
        # x = self.prelu(self.skff_last([x, residual, skout]))
        return x


class ConvMLPnet(nn.Module):
    def __init__(self,
                 blocks,
                 dims,
                 mlp_ratios,
                 channels=64,
                 n_conv_blocks=3,
                 *args, **kwargs):
        super(ConvMLPnet, self).__init__()
        assert len(blocks) == len(dims) == len(mlp_ratios), \
            f"blocks, dims and mlp_ratios must agree in size, {len(blocks)}, {len(dims)} and {len(mlp_ratios)} passed."

        self.tokenizer = ConvTokenizer(embedding_dim=channels)
        self.conv_stages = ConvStage(n_conv_blocks,
                                     embedding_dim_in=channels,
                                     hidden_dim=dims[0],
                                     embedding_dim_out=dims[0])
        self.first = nn.Conv2d(3, channels, 3, 1, 1)
        self.stages = nn.ModuleList()
        for i in range(blocks[0]):
            stage = BasicStage(num_blocks=blocks[1],
                               embedding_dims=dims[0:2],
                               mlp_ratio=mlp_ratios[0],
                               stochastic_depth_rate=0.1,
                               downsample=False
                               # downsample=(i + 1 < len(blocks))
                               )
            self.stages.append(stage)
        self.out = nn.Sequential(
            nn.Conv2d(dims[-1], dims[-1] // 4, 3, 1, 1),
            nn.Conv2d(dims[-1] // 4, 3, 3, 1, 1)
        )
        self.out_last = nn.Sequential(nn.Conv2d(dims[0], dims[0], 3, 1, 1), nn.Conv2d(dims[0], dims[0], 3, 1, 1))
        self.skff = SKFF(dims[0], blocks[0] + 1)
        # self.skout = nn.Conv2d(dims[0], 3, 3, 1, 1)
        # self.skff_last = SKFF(dims[0], 3)
        self.prelu = nn.PReLU()
        # self.information_pool = nn.Conv2d((len(blocks) + 1) * dims[0], 3, 3, 1, 1)
        # self.apply(self.init_weight)

    def forward(self, x):
        # x = self.tokenizer(x)
        # x = self.conv_stages(x)
        x = self.first(x)
        global_residual = x
        # x = x.permute(0, 2, 3, 1)  # NCHW->NHWC
        # residual = x
        out = []
        # x = self.stages(x)
        for stage in self.stages:
            x = stage(x)
            out.append(x)
            # out = torch.cat((out, x), dim=1)
            # x = residual + x
        # x = x + residual
        # x = x.permute(0, 3, 1, 2)
        # B, _, _, C = x.shape
        # x = self.out(x)
        out.append(global_residual)
        x = self.out_last(self.prelu(self.skff(out)))
        # skout = self.skout(self.skff(out))  # + self.information_pool(out))
        # x = self.out_last(self.prelu(self.skff_last([x, skout, global_residual])))

        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, (nn.Linear, nn.Conv1d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)


class ConvMLP(nn.Module):
    def __init__(self, mode=None):
        super(ConvMLP, self).__init__()
        if mode == 'm':
            self.model = self.convmlp_m()
        elif mode == 'l':
            self.model = self.convmlp_l()
        elif mode == 's':
            self.model = self.convmlp_s()
        else:
            raise Exception('error mode in (s,l,m)')
        self.upsample = Upsample(4, 64)
        self.conv = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1),
                                  nn.Conv2d(64, 3, 3, 1, 1),
                                  # nn.Conv2d(3, 3, 3, 1, 1),
                                  )

    def forward(self, x):
        out = self.model(x)
        out = self.conv(self.upsample(out)) + nn.functional.interpolate(x, scale_factor=4, mode='bicubic',
                                                                        align_corners=True)
        return out

    def _convmlp(self, blocks, dims, mlp_ratios, n_conv_blocks=3, *args, **kwargs):
        model = ConvMLPnet(blocks=blocks, dims=dims, mlp_ratios=mlp_ratios, n_conv_blocks=n_conv_blocks, *args,
                           **kwargs)
        return model

    def convmlp_s(self, *args, **kwargs):
        return self._convmlp(blocks=[2, 4, 2], mlp_ratios=[2, 2, 2], dims=[128, 256, 512],
                             channels=64, n_conv_blocks=2, *args, **kwargs)

    def convmlp_m(self, *args, **kwargs):  # server
        return self._convmlp(blocks=[2, 6, 1, 1], mlp_ratios=[2, 2, 2, 2], dims=[64, 64, 64, 64], channels=64,
                             n_conv_blocks=2, *args, **kwargs)

    def convmlp_l(self, *args, **kwargs):
        return self._convmlp(blocks=[4, 8, 3], mlp_ratios=[3, 3, 3], dims=[192, 384, 768],
                             channels=96, n_conv_blocks=3, *args, **kwargs)

# from torchsummary import summary
#
# if __name__ == '__main__':
#     model = ConvMLP().cuda()
#     summary(model, (3, 24, 24))
