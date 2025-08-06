"""
iFormer: Efficient Hierarchical Attention Network
根据 iformer.arch 文件构建的简洁版本模型结构

基于原始iFormer架构重新实现的简化版本，使用timm库提供的标准组件

模型特点:
1. 4阶段层次化架构: Stage0/1使用纯卷积，Stage2/3混合注意力机制
2. 渐进式下采样: 3→16→dims[0]→dims[1]→dims[2]→dims[3]  
3. 动态块构建: 根据depths参数自适应构建各阶段的块数量
4. 多尺度模型: 提供tiny/small/base/medium四种规模

使用示例:
    import torch
    from models.aha import iformer_base
    
    model = iformer_base(num_classes=1000)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)  # 输出: [1, 1000]

模型规模对比:
- tiny:   16→32→64→128   (0.7M参数, ~2.7MB)
- small:  24→48→96→192   (1.5M参数, ~5.8MB) 
- base:   32→64→128→256  (2.9M参数, ~10.9MB)
- medium: 48→96→192→384  (8.6M参数, ~32.7MB)
"""

from enum import CONFORM
from functools import partial
import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_, make_divisible
from models.ops.conv import ConvNorm, RepConv, Scale
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
        self.layer_scale = Scale(dim=dim, init_value=ls_init) if ls_init > 0 else nn.Identity()
        self.drop_path = DropPath(drop_prob=dp) if dp > 0. else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.layer_scale(self.m(x)))
    
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
    def __init__(self, dim, attn_type, attn_reduce, attn_norm, attn_gate, la_kernel):
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
        self.gate_act = get_act(attn_gate)
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
        x = self.attn(q, k, v) * self.gate_act(g)
        x = x.reshape(B, self.d, H, W)
        x = self.proj(x)
        return x


class AttnBlock(nn.Module):
    """注意力块 - 包含SHMA和MLP"""
    def __init__(self, dim, attn_type, attn_reduce, attn_gate, la_kernel,
                 mlp_ratio, mlp_act, use_glu, ls_init, dp):
        super().__init__()
        self.token_mixer = 
        
    def forward(self, x):
        return self.token_channel_mixer(x)
    
class SHMABlock(nn.Module):
    """SHMA注意力块"""
    def __init__(self, dim, num_heads=1, drop_path=0., layer_scale_init_value=1e-6, **kwargs):
        super().__init__()
        self.token_channel_mixer = Residual(
            SHMA(dim, num_heads=num_heads, **kwargs), 
            drop_path=drop_path,
            layer_scale_init_value=layer_scale_init_value,
            dim=dim
        )
        
    def forward(self, x):
        return self.token_channel_mixer(x)


class FFN2d(nn.Module):
    """2D前馈网络块"""
    def __init__(self, dim, mlp_ratio=2.0, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        
        ffn = nn.Sequential(
            Conv2d_BN(dim, hidden_dim, 1),
            nn.GELU(),
            Conv2d_BN(hidden_dim, dim, 1),
        )
        
        self.channel_mixer = Residual(
            ffn, 
            drop_path=drop_path,
            layer_scale_init_value=layer_scale_init_value,
            dim=dim
        )
        
    def forward(self, x):
        return self.channel_mixer(x)


class BasicBlock(nn.Module):
    """基础块包装器 - 根据架构文件自动选择块类型"""
    def __init__(self, block):
        super().__init__()
        self.block = block
        
    def forward(self, x):
        return self.block(x)


class BN_Linear(nn.Module):
    """批归一化+线性层"""
    def __init__(self, in_features, out_features, bias=True, std=0.02):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.l = nn.Linear(in_features, out_features, bias=bias)
        trunc_normal_(self.l.weight, std=std)
        if bias:
            nn.init.constant_(self.l.bias, 0)
        
    def forward(self, x):
        return self.l(self.bn(x))


class Classifier(nn.Module):
    """分类器模块 - 全局平均池化 + BN_Linear"""
    def __init__(self, in_features, num_classes=1000):
        super().__init__()
        self.classifier = BN_Linear(in_features, num_classes) if num_classes > 0 else nn.Identity()
        
    def forward(self, x):
        # 全局平均池化
        x = x.mean(dim=(2, 3))  # B, C, H, W -> B, C
        return self.classifier(x)


class iFormer(nn.Module):
    """
    iFormer: Efficient Hierarchical Attention Network
    基于 iformer.arch 重新实现的简洁版本
    
    Args:
        dims: 各阶段的通道数, 默认 [32, 64, 128, 256]
        depths: 各阶段的块数量, 默认 [2, 2, 16, 6] 
        num_classes: 分类数量, 默认 1000
        drop_path_rate: drop path 比率, 默认 0.0
        layer_scale_init_value: layer scale 初始值, 默认 1e-6
    """
    def __init__(self, 
                 dims=None, 
                 depths=None, 
                 num_classes=1000, 
                 drop_path_rate=0.0,
                 layer_scale_init_value=1e-6):
        super().__init__()
        self.num_classes = num_classes
        
        # 默认配置 - 基于 iformer.arch 的结构
        if dims is None:
            dims = [32, 64, 128, 256]
        if depths is None:
            depths = [2, 2, 16, 6]
        
        # 构建下采样层 - 参考架构文件的downsample_layers
        self.downsample_layers = nn.ModuleList()
        
        # Stage 0 Stem: 3 -> 16 -> (expand to 64) -> dims[0]
        stem = nn.Sequential(
            Conv2d_BN(3, 16, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            EdgeResidual(16, dims[0], exp_kernel_size=5, stride=2, exp_ratio=4.0)
        )
        self.downsample_layers.append(stem)
        
        # Stage 1-3: 逐层下采样 32->64, 64->128, 128->256
        for i in range(3):
            downsample = nn.Sequential(
                Conv2d_BN(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1)
            )
            self.downsample_layers.append(downsample)
        
        # 计算每个块的drop path rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        # 构建各个stage - 基于架构文件的具体结构
        self.stages = nn.ModuleList()
        
        # Stage 0: 2个ConvBlock (dims[0]=32)
        stage0_blocks = []
        for j in range(depths[0]):
            block = BasicBlock(ConvBlock(
                dims[0], 
                kernel_size=7, 
                mlp_ratio=3, 
                drop_path=dpr[cur + j],
                layer_scale_init_value=layer_scale_init_value
            ))
            stage0_blocks.append(block)
        self.stages.append(nn.Sequential(*stage0_blocks))
        cur += depths[0]
        
        # Stage 1: 2个ConvBlock (dims[1]=64)
        stage1_blocks = []
        for j in range(depths[1]):
            block = BasicBlock(ConvBlock(
                dims[1], 
                kernel_size=7, 
                mlp_ratio=3, 
                drop_path=dpr[cur + j],
                layer_scale_init_value=layer_scale_init_value
            ))
            stage1_blocks.append(block)
        self.stages.append(nn.Sequential(*stage1_blocks))
        cur += depths[1]
        
        # Stage 2: 混合架构 (dims[2]) - 根据depths[2]动态构建
        stage2_blocks = []
        remaining_blocks = depths[2]
        
        # 如果有足够的块，先添加6个ConvBlock，否则全部用ConvBlock
        conv_blocks_count = min(6, remaining_blocks)
        for j in range(conv_blocks_count):
            block = BasicBlock(ConvBlock(
                dims[2], 
                kernel_size=7, 
                mlp_ratio=3, 
                drop_path=dpr[cur + j],
                layer_scale_init_value=layer_scale_init_value
            ))
            stage2_blocks.append(block)
        
        remaining_blocks -= conv_blocks_count
        block_idx = conv_blocks_count
        
        # 添加注意力块组 (RepCPE + SHMA + FFN)
        while remaining_blocks >= 3:
            # RepCPE
            stage2_blocks.append(BasicBlock(RepCPE(
                dims[2], 
                kernel_size=3, 
                drop_path=dpr[cur + block_idx],
                layer_scale_init_value=layer_scale_init_value
            )))
            # SHMA
            stage2_blocks.append(BasicBlock(SHMABlock(
                dims[2], 
                num_heads=1, 
                drop_path=dpr[cur + block_idx + 1],
                layer_scale_init_value=layer_scale_init_value
            )))
            # FFN
            stage2_blocks.append(BasicBlock(FFN2d(
                dims[2], 
                mlp_ratio=2, 
                drop_path=dpr[cur + block_idx + 2],
                layer_scale_init_value=layer_scale_init_value
            )))
            
            remaining_blocks -= 3
            block_idx += 3
        
        # 如果还有剩余的块，用ConvBlock填充
        for j in range(remaining_blocks):
            block = BasicBlock(ConvBlock(
                dims[2], 
                kernel_size=7, 
                mlp_ratio=3, 
                drop_path=dpr[cur + block_idx + j],
                layer_scale_init_value=layer_scale_init_value
            ))
            stage2_blocks.append(block)
        
        self.stages.append(nn.Sequential(*stage2_blocks))
        cur += depths[2]
        
        # Stage 3: 注意力阶段 (dims[3]) - 根据depths[3]动态构建
        stage3_blocks = []
        remaining_blocks = depths[3]
        block_idx = 0
        
        # 添加注意力块组 (RepCPE + SHMA + FFN)
        while remaining_blocks >= 3:
            # RepCPE
            stage3_blocks.append(BasicBlock(RepCPE(
                dims[3], 
                kernel_size=3, 
                drop_path=dpr[cur + block_idx],
                layer_scale_init_value=layer_scale_init_value
            )))
            # SHMA  
            stage3_blocks.append(BasicBlock(SHMABlock(
                dims[3], 
                num_heads=1, 
                drop_path=dpr[cur + block_idx + 1],
                layer_scale_init_value=layer_scale_init_value
            )))
            # FFN
            stage3_blocks.append(BasicBlock(FFN2d(
                dims[3], 
                mlp_ratio=2, 
                drop_path=dpr[cur + block_idx + 2],
                layer_scale_init_value=layer_scale_init_value
            )))
            
            remaining_blocks -= 3
            block_idx += 3
        
        # 如果还有剩余的块，用ConvBlock填充
        for j in range(remaining_blocks):
            block = BasicBlock(ConvBlock(
                dims[3], 
                kernel_size=7, 
                mlp_ratio=3, 
                drop_path=dpr[cur + block_idx + j],
                layer_scale_init_value=layer_scale_init_value
            ))
            stage3_blocks.append(block)
        
        self.stages.append(nn.Sequential(*stage3_blocks))
        
        # 分类器 - 参考架构文件的classifier部分
        self.classifier = Classifier(dims[-1], num_classes)
        
        # 权重初始化
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
        """特征提取"""
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x
    
    def forward(self, x):
        """前向传播"""
        x = self.forward_features(x)
        x = self.classifier(x)
        return x

# 模型工厂函数
def iformer_tiny(num_classes=1000, **kwargs):
    """iFormer微型版本 - 适用于资源受限场景"""
    model = iFormer(
        dims=[16, 32, 64, 128],
        depths=[1, 1, 8, 2],
        num_classes=num_classes,
        **kwargs
    )
    return model


def iformer_small(num_classes=1000, **kwargs):
    """iFormer小版本 - 性能与效率的平衡"""
    model = iFormer(
        dims=[24, 48, 96, 192],
        depths=[2, 2, 12, 4],
        num_classes=num_classes,
        **kwargs
    )
    return model


def iformer_base(num_classes=1000, **kwargs):
    """iFormer基础版本 - 与原始架构完全对应"""
    model = iFormer(
        dims=[32, 64, 128, 256],
        depths=[2, 2, 16, 6],
        num_classes=num_classes,
        **kwargs
    )
    return model


def iformer_medium(num_classes=1000, **kwargs):
    """iFormer中等版本 - 更强的表现力"""
    model = iFormer(
        dims=[48, 96, 192, 384],
        depths=[3, 3, 20, 8],
        num_classes=num_classes,
        **kwargs
    )
    return model


if __name__ == "__main__":
    # 测试模型
    print("=" * 50)
    print("Testing iFormer models...")
    print("=" * 50)
    
    # 测试输入
    x = torch.randn(2, 3, 224, 224)
    print(f"Input shape: {x.shape}")
    
    # 测试不同大小的模型
    models = {
        'tiny': iformer_tiny(),
        'small': iformer_small(), 
        'base': iformer_base(),
        'medium': iformer_medium()
    }
    
    for name, model in models.items():
        try:
            model.eval()
            with torch.no_grad():
                y = model(x)
            
            # 计算参数量
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"\n{name.upper()} model:")
            print(f"  Output shape: {y.shape}")
            print(f"  Parameters: {params:,}")
            print(f"  Model size: ~{params * 4 / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            print(f"\n{name.upper()} model FAILED: {e}")
    
    print("\n" + "=" * 50)
    print("Testing completed!")
    print("=" * 50)
