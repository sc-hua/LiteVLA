import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as cp
from timm.layers import make_divisible
from timm.utils.model import reparameterize_model

from models.ops.conv import ConvNorm, RepConv, ConvGate, Scale
from models.ops.lib import GLOBAL_EPS, use_linear
from models.ops.norm_act import get_norm, get_act


class GatedAggBlock(nn.Module):
    def __init__(self, dim, k, act, n=2, e=0.5):
        super().__init__()
        d = int(dim * e)
        d_out = (n + 1) * d
        
        self.x = ConvNorm(dim, d)
        self.act = get_act(act)
        self.agg = nn.ModuleList(RepConv(d, d, k=k, g=d) for _ in range(n))
        self.g = ConvGate(dim, d_out, act=nn.Sigmoid)
        self.out = ConvNorm(d_out, dim, bn_w_init=0.)
        self.rs = Scale(dim)

    def forward(self, x):
        y = [self.x(x)]
        for agg in self.agg:
            z = y[-1]
            y.append(z + self.act(agg(z)))
        y = torch.cat(y, 1) * self.g(x)
        return self.out(y) + self.rs(x)
    

class ELA(nn.Module):
    def __init__(self, dim, attn_ratio, attn_kernel, attn_norm, dwc_kernel):
        super().__init__()        
        d = make_divisible(int(dim * attn_ratio), 8)
        d_qk, d_v = d, 2 * d
        slices = [d_qk, d_qk, d_v]
        attn_dim = sum(slices)
        self.slices = slices
        
        self.eps = GLOBAL_EPS
        self.kernel = get_act(attn_kernel)
        self.proj = ConvNorm(dim, attn_dim)
        self.dwc = RepConv(attn_dim, attn_dim, k=dwc_kernel, g=attn_dim, res=True)
        self.g = ConvGate(attn_dim, d_v, act=nn.Sigmoid, conv1d=True)
        self.attn_norm = get_norm(attn_norm, d_v, w_init=0.)
        self.scale_v = Scale(d_v, shape=(1, d_v, 1))
        self.out = ConvNorm(d_v, dim, bn_w_init=0.)
    
    def attn(self, q, k, v):
        q, kT = self.kernel(q), self.kernel(k).transpose(-1,-2)
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
        B, _, H, W = x.shape
        x = self.dwc(self.proj(x)).flatten(2)  # local aggregation
        q, k, v = x.split(self.slices, dim=1)
        attn = self.attn(q, k, v) * self.g(x)  # input-dependent gating
        attn = self.attn_norm(attn) + self.scale_v(v)  # res for stability
        return self.out(attn.reshape(B, -1, H, W))
    

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio, act):
        super().__init__()
        hid = make_divisible(int(dim * mlp_ratio), 8)
        self.fc1 = ConvNorm(dim, hid)
        self.act = get_act(act)
        self.fc2 = ConvNorm(hid, dim, bn_w_init=0.)
        
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class ELABlock(nn.Module):
    def __init__(self, dim, attn_ratio, attn_kernel, attn_norm, dwc_kernel, mlp_ratio, act):
        super().__init__()
        self.ela = ELA(dim, attn_ratio, attn_kernel, attn_norm, dwc_kernel)
        self.rs1 = Scale(dim)
        self.mlp = MLP(dim, mlp_ratio, act)
        self.rs2 = Scale(dim)

    def forward(self, x):
        x = self.rs1(x) + self.ela(x)
        return self.rs2(x) + self.mlp(x)


class MBConv(nn.Module):
    def __init__(self, inp, oup, exp, k, s, act, fuse=False):
        super().__init__()
        hid = int(oup * exp)
        res = (s == 1 and inp == oup)
        if fuse:
            ops = [RepConv(inp, hid, k, s, res=res)]  # c,h,w -> hid,h/2,w/2
        else:
            ops = [
                ConvNorm(inp, hid), get_act(act),  # c -> hid
                RepConv(hid, hid, k, s, g=hid, res=res) # h,w -> h/2,w/2
            ]
        ops += [get_act(act), ConvNorm(hid, oup)]  # hid -> 2c 
        self.conv = nn.Sequential(*ops)

    def forward(self, x): 
        return self.conv(x) # b,c,h,w -> b,2c,h/2,w/2
    

class BasicLayer(nn.Module):
    def __init__(self, inp, oup, depth, ds_exp, ds_kernel, ds_fuse, act, dwc_kernel, block_type, 
                 attn_ratio, attn_kernel, attn_norm, mlp_ratio, use_checkpoint):
        super().__init__()
        # downsample
        blocks = [MBConv(inp=inp, oup=oup, exp=ds_exp, k=ds_kernel, s=2, act=act, fuse=ds_fuse)]
        
        # blocks
        assert block_type in ['VLA', 'GAB'], f'block_type must be VLA or GAB, but got {block_type}'
        if block_type == 'GAB':
            blocks += [GatedAggBlock(dim=oup, k=dwc_kernel, act=act) for _ in range(depth)]
        else:
            blocks += [ELABlock(
                dim=oup, attn_ratio=attn_ratio, attn_kernel=attn_kernel, attn_norm=attn_norm,
                dwc_kernel=dwc_kernel, mlp_ratio=mlp_ratio, act=act) for _ in range(depth)]
        self.blocks = nn.ModuleList(blocks)
        self.cp = use_checkpoint

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x) if not self.cp else cp(blk, x)
        return x


class Classifier(nn.Module):
    def __init__(self, num_features, num_classes, head_norm, distillation=False):
        super().__init__()
        self.distillation = distillation
        self.norm = get_norm(head_norm, num_features)
        self.head = nn.Linear(num_features, num_classes)
        if distillation:
            self.dist_head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # pool -> norm -> linear
        x = x.mean((2,3), False)  # b,c,h,w -> b,c
        x = self.norm(x)
        if self.distillation and self.dist_head is not None:
            x = self.head(x), self.dist_head(x)
            if not self.training:
                x = (x[0] + x[1]) / 2.
        else:
            x = self.head(x)
        return x
    
    @torch.no_grad()
    def reparameterize(self):
        h = reparameterize_model(self.head)
        if self.distillation:
            d = reparameterize_model(self.dist_head)
            h.weight = (h.weight + d.weight) / 2
            h.bias = (h.bias + d.bias) / 2
            self.dist_head = None
        self.head = h


class LiteVLA(nn.Module):
    def __init__(self, inp=3, num_classes=1000, dims=(48, 96, 192, 384), 
                 depths=(2, 2, 11, 3), block_types=('GAB', 'GAB', 'VLA', 'VLA'), 
                 stem_kernel=5, ds_exp=4, ds_kernel=3, ds_fuse="ff--", act='silu', dwc_kernel=5,
                 attn_ratio=1/8, attn_kernel='elu1', attn_norm='mrms', mlp_ratio=3, head_norm='mrms',
                 use_checkpoint=False, distillation=False,  # for training
                 backbone=False, out_indices=None, pretrained=None,  # for backbone
                 **kwargs):
        super().__init__()
        ablation = kwargs.pop('ablation', '')
        if ablation != '' and len(ablation) > 0:
            raise ValueError(f"ablation: {kwargs['ablation']}, please refer to the ablation version of LiteVLA")
        if isinstance(dims, int):
            dims = [dims * 2 ** i for i in range(len(depths))]
        ds_fuse = [f == 'f' for f in ds_fuse]
        
        # Stem
        dims = [dims[0] // 2, *dims]
        self.stem = nn.Sequential(RepConv(inp, dims[0], k=stem_kernel, s=2), get_act(act))
        
        # build layers [ DownSample + Blocks ] x 4
        self.layers = nn.ModuleList([BasicLayer(
            inp=dims[i], oup=dims[i+1], depth=depths[i], ds_exp=ds_exp, ds_kernel=ds_kernel, 
            ds_fuse=ds_fuse[i], act=act, dwc_kernel=dwc_kernel, block_type=block_types[i], 
            attn_ratio=attn_ratio, attn_kernel=attn_kernel, attn_norm=attn_norm,
            mlp_ratio=mlp_ratio, use_checkpoint=use_checkpoint, 
        ) for i in range(len(depths))])
        dims.pop(0)
        
        # Classifier
        self.head = Classifier(dims[-1], num_classes, head_norm, distillation)

        # for downstream
        self.load_pretrained(pretrained)
        self.out_indices = out_indices
        self.backbone = backbone
        if self.backbone or self.out_indices is not None:
            delattr(self, 'head')
        if self.out_indices is not None:
            self = nn.SyncBatchNorm.convert_sync_batchnorm(self)  # for batchnorm
            assert isinstance(out_indices, (tuple, list)), 'out_indices should be tuple or list'
            assert max(self.out_indices) < len(self.layers), 'out_indices should < len(layers)'
            assert len(self.out_indices) > 0, 'len(out_indices) sould > 0'

    def forward_features(self, x):
        out=[]
        x = self.stem(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.out_indices is not None and i in self.out_indices:
                out.append(x)
        return x if self.out_indices is None else out

    def forward(self, x):
        x = self.forward_features(x)
        if self.backbone or self.out_indices is not None:
            return x
        return self.head(x)
                    
    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"(LiteVLA): Loading ckpt from {ckpt} ...")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(f"(LiteVLA): incompatibleKeys: {incompatibleKeys}")
        except Exception as e:
            print(f"(LiteVLA): Failed loading checkpoint form {ckpt}: {e}")

