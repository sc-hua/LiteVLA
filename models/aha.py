import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_, make_divisible
try:
    from models.ops.conv import ConvNorm, BNLinear, RepConv, Scale
    from models.ops.norm_act import get_act, get_norm
    from models.lib import use_linear
except:
    # allow running this file directly: `python models/aha.py`
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from models.ops.conv import ConvNorm, BNLinear, RepConv, Scale
    from models.ops.norm_act import get_act, get_norm
    from models.lib import use_linear


class EdgeResidual(nn.Sequential):
    """FusedIB模块 - MobileNetV4中的融合反转瓶颈"""
    def __init__(self, inp, oup, k, s, exp, act):
        super().__init__()
        mid = int(inp * exp)
        self.add_module('conv_exp_bn1', ConvNorm(inp, mid, k=k, s=s))
        self.add_module('act', get_act(act))
        self.add_module('conv_pw_bn2', ConvNorm(mid, oup))


class Residual(nn.Module):
    """残差连接模块，支持DropPath和LayerScale"""
    def __init__(self, m, dim, ls_init, dp):
        super().__init__()
        self.m = m
        self.layer_scale = Scale(dim=dim, init_value=ls_init) if ls_init > 0 else None
        self.drop_path = DropPath(drop_prob=dp) if dp > 0. else None

    def forward(self, x):
        y = self.m(x)
        if self.layer_scale is not None:
            y = self.layer_scale(y)
        if self.drop_path is not None:
            y = self.drop_path(y)
        return x + y

    @torch.no_grad()
    def fuse(self):
        # HSC: Need to implement fusion logic if applicable
        return self


class MLP(nn.Module):
    def __init__(self, dim, exp, mlp_act, use_glu):
        super().__init__()
        exp = exp * 2 / 3 if use_glu else exp
        hid = make_divisible(dim * exp, divisor=8)
        self.use_glu = use_glu
        self.fc1 = ConvNorm(dim, hid)
        if use_glu:
            self.gate = ConvNorm(dim, hid)
        self.act = get_act(mlp_act)
        self.fc2 = ConvNorm(hid, dim)
        
    def forward(self, x):
        if self.use_glu:
            return self.fc2(self.fc1(x) * self.act(self.gate(x)))
        return self.fc2(self.act(self.fc1(x)))


class ConvBlock(nn.Module):
    """卷积块 - DWConv + MLP"""
    def __init__(self, dim, k, mlp_ratio, use_glu, mlp_act, ls_init, dp):
        super().__init__()
        self.dwc = RepConv(inp=dim, oup=dim, k=k, g=dim, res=True)
        mlp = MLP(dim=dim, exp=mlp_ratio, mlp_act=mlp_act, use_glu=use_glu)
        self.mlp = Residual(m=mlp, dim=dim, ls_init=ls_init, dp=dp)

    def forward(self, x):
        x = self.dwc(x)
        x = self.mlp(x)
        return x


class SHMA(nn.Module):
    """单头调和注意力 (Single Head Modulated Attention)"""
    def __init__(self, dim, attn_type, attn_reduce, attn_norm, attn_gate_act, la_kernel):
        super().__init__()
        self.dim = dim
        self.reduce = attn_reduce
        self.d = make_divisible(dim / attn_reduce, divisor=8)
        self.attn_type = attn_type
        
        if attn_type == 'SA':
            self.attn = self.sa
            self.scale = self.d ** -0.5
        elif attn_type == 'LA':
            self.attn = self.la
            self.kernel = get_act(la_kernel)
            self.eps = 5e-4
        else:
            raise ValueError("attn_type must be 'SA' or 'LA'")
        
        self.qkvg = ConvNorm(dim, 4 * self.d)
        self.gate_act = get_act(attn_gate_act)
        self.attn_norm = get_norm(attn_norm, self.d)
        self.proj = ConvNorm(self.d, dim)
    
    def sa(self, q, k, v):
        """Softmax Attention"""
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return attn @ v

    def la(self, q, k, v):
        """Linear Attention"""
        q, kT = self.kernel(q), self.kernel(k).transpose(-1,-2)
        
        # use `scale`, `eps` and `.mean()` for numerical stability
        scale = kT.shape[-2] ** -0.5
        if use_linear(q, v):
            z = kT.mean(-2, True) @ q + self.eps
            v, kT = v * scale, kT * scale
            attn = (v @ kT) @ q
        else:
            kq = kT @ q  # B, Nvk, Nq
            z = kq.mean(-2, True) + self.eps
            v, kq = v * scale, kq * scale
            attn = v @ kq
        return attn / z

    def forward(self, x):
        B, C, H, W = x.shape
        q,k,v,g = self.qkvg(x).flatten(2).chunk(4, dim=1)
        x = self.attn(q, k, v)
        x = self.attn_norm(x)
        x = x * self.gate_act(g)
        x = x.reshape(B, self.d, H, W)
        x = self.proj(x)
        return x


class AttnBlock(nn.Module):
    """注意力块 - 包含CPE,SHMA和MLP"""
    def __init__(self, dim, cpe_k, attn_type, attn_reduce, attn_norm, attn_gate_act, la_kernel,
                 mlp_ratio, mlp_act, use_glu, ls_init, dp):
        super().__init__()
        self.cpe = RepConv(inp=dim, oup=dim, k=cpe_k, g=dim, res=True)
        shma = SHMA(
            dim=dim, attn_type=attn_type, attn_reduce=attn_reduce,
            attn_norm=attn_norm, attn_gate_act=attn_gate_act, la_kernel=la_kernel
        )
        mlp = MLP(dim=dim, exp=mlp_ratio, mlp_act=mlp_act, use_glu=use_glu)
        self.shma = Residual(m=shma, dim=dim, ls_init=ls_init, dp=dp)
        self.mlp = Residual(m=mlp, dim=dim, ls_init=ls_init, dp=dp)

    def forward(self, x):
        x = self.cpe(x)
        x = self.shma(x)
        x = self.mlp(x)
        return x


class AHA(nn.Module):
    """AHA: Adaptive Hybrid Attention for Mobile Vision Applications"""
    def __init__(self, inp=3, 
                 num_classes=1000, 
                 dims=(48, 96, 192, 384),
                 depths=(2, 2, 12, 4),
                 mlp_ratios=(4, 4, 3, 3),
                 act='silu',
                 attn_start_id=2+2+8, # 第三个 Stage 的第8个块开始
                 drop_path_rate=0.,
                 layer_scale_init_value=0.,
                 **kwargs):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        downsample_kernels=(5, 3, 3, 3) # check if 5 will increase acc.
        conv_kernels = (3, 3, 3, 3)
        cpe_kernels = (3, 3, 3, 3)
        attn_gate_act = 'sigmoid'
        use_glu = False
        mlp_act = act

        self.num_classes = num_classes
        self.stem = nn.Sequential(
            ConvNorm(inp=inp, oup=dims[0] // 2, k=downsample_kernels[0], s=2), 
            get_act(act),
        )
        
        cur = 0  # current block id
        inp = dims[0] // 2
        self.stages = nn.ModuleList()
        for idx in range(len(dims)):
            dim = dims[idx]
            depth = depths[idx]
            mlp_ratio = mlp_ratios[idx]
            down_kernel = downsample_kernels[idx]
            conv_kernel = conv_kernels[idx]
            cpe_kernel = cpe_kernels[idx]

            downsample = (
                EdgeResidual(inp=inp, oup=dim, k=down_kernel, s=2, exp=mlp_ratio, act=act)
                if idx == 0 else ConvNorm(inp=inp, oup=dim, k=down_kernel, s=2)
            )
            blocks = [downsample]
            for i in range(depth):
                if cur < attn_start_id:
                    blocks.append(ConvBlock(
                        dim=dim, k=conv_kernel, mlp_ratio=mlp_ratio, use_glu=use_glu, 
                        mlp_act=mlp_act, ls_init=layer_scale_init_value, dp=dpr[cur]
                    ))
                else:
                    attn_type = "SA" if i == depth - 1 else "LA"
                    blocks.append(AttnBlock(
                        dim=dim, cpe_k=cpe_kernel, attn_type=attn_type,
                        attn_reduce=2, attn_norm='gn1', attn_gate_act=attn_gate_act,
                        la_kernel='elu1', mlp_ratio=mlp_ratio, mlp_act=mlp_act,
                        use_glu=use_glu, ls_init=layer_scale_init_value, dp=dpr[cur]
                    ))
                cur += 1
            blocks = nn.ModuleList(blocks)
            self.stages.append(blocks)
            inp = dim

        self.classifier = BNLinear(dims[-1], num_classes)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            for block in stage:
                x = block(x)
        return x
    
    def forward(self, x):
        """前向传播"""
        x = self.forward_features(x)
        x = x.mean((-2, -1))
        x = self.classifier(x)
        return x

def aha_tiny(num_classes=1000, **kwargs):
    """AHA微型版本 - 适用于资源受限场景"""
    raise NotImplementedError


def aha_small(num_classes=1000, **kwargs):
    """AHA小版本 - 性能与效率的平衡"""
    raise NotImplementedError


def aha_medium(num_classes=1000, **kwargs):
    """AHA中等版本 - 更强的表现力"""
    model = AHA(
        dims=(48, 96, 192, 384),
        depths=(2, 2, 12, 4),
        mlp_ratios=(4, 4, 3, 3),
        attn_start_id=2+2+8,  # 第三个 Stage 的第8个块开始
        num_classes=num_classes,
        **kwargs
    )
    return model


def aha_large(num_classes=1000, **kwargs):
    """AHA大型版本 - 最高性能"""
    raise NotImplementedError


def get_aha_model(version, **kwargs):
    if version in ['tiny', 't']:
        return aha_tiny(**kwargs)
    elif version in ['small', 's']:
        return aha_small(**kwargs)
    elif version in ['medium', 'm']:
        return aha_medium(**kwargs)
    elif version in ['large', 'l']:
        return aha_large(**kwargs)
    else:
        raise NotImplementedError(f"Unknown AHA version: {version}")


if __name__ == "__main__":
    # 测试模型
    print("=" * 50)
    print("Testing iFormer models...")
    print("=" * 50)
    
    # 测试输入
    x = torch.randn(1, 3, 224, 224)
    print(f"Input shape: {x.shape}")
    
    m = aha_small()
    
    # get model, params and flops
    from timm.utils.model import reparameterize_model
    m = reparameterize_model(m)
    print(m)
    
    from thop import profile
    flops, params = profile(m, inputs=(x,))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Params: {params / 1e6:.2f} M")

    m.eval()
    with torch.no_grad():
        y = m(x)
    print(f"Output shape: {y.shape}")
    
    print("\n" + "=" * 50)
    print("Testing completed!")
    print("=" * 50)
