import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm
from timm.layers import make_divisible, DropPath
from timm.utils.model import reparameterize_model

from models.ops.conv import ConvNorm, RepConv, ConvGate, Scale
from models.lib import GLOBAL_EPS, use_linear
from models.ops.norm_act import get_norm, get_act


class GatedAggBlock(nn.Module):
    def __init__(self, dim, k, act, drop_path, n=2, e=0.5, **kwargs):
        super().__init__()
        d = int(dim * e)
        d_out = (n + 1) * d
        
        self.x = ConvNorm(dim, d)
        self.act = get_act(act)
        no_rep = kwargs.get('no_rep', False)
        self.agg = nn.ModuleList(RepConv(d, d, k=k, g=d, use_rep=not no_rep) for _ in range(n))
        
        self.no_gate = kwargs.get('no_gate', False)
        if not self.no_gate:
            gate_act = kwargs.get('gate_act', 'sigmoid')
            gate_act = get_act(gate_act)
            self.g = ConvGate(dim, d_out, act=gate_act)
        self.out = ConvNorm(d_out, dim, bn_w_init=0.)
        self.dp = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.rs = Scale(dim)

    def forward(self, x):
        y = [self.x(x)]
        for agg in self.agg:
            z = y[-1]
            y.append(z + self.act(agg(z)))
        y = torch.cat(y, 1)
        if not self.no_gate:
            y = y * self.g(x)
        y = self.out(y)
        return self.rs(x) + self.dp(y)
    

class ELA(nn.Module):
    def __init__(self, dim, attn_ratio, attn_kernel, attn_norm, dwc_kernel, **kwargs):
        super().__init__()        
        d = make_divisible(int(dim * attn_ratio), 8)
        d_qk, d_v = d, 2 * d
        slices = [d_qk, d_qk, d_v]
        attn_dim = sum(slices)
        self.slices = slices
        
        self.proj = ConvNorm(dim, attn_dim)
        
        # ablation: repconv or not
        no_rep = kwargs.get('no_rep', False)
        self.dwc = RepConv(attn_dim, attn_dim, k=dwc_kernel, g=attn_dim, res=True, use_rep=not no_rep)
        
        # ablation: use attn or not
        # ablation: linear_attn or softmax_attn
        self.no_attn = kwargs.get('no_attn', False)
        if self.no_attn:
            inner_dim = attn_dim
            conv_1d = False
            attn_type = 'None'
        else:
            inner_dim = d_v
            conv_1d = True
            use_sa = kwargs.get('use_sa', False)
            if use_sa:
                self.attn = self.softmax_attn
                self.scale = d_qk ** -0.5
                attn_type = 'Softmax'
            else:
                self.attn = self.linear_attn
                self.eps = GLOBAL_EPS
                self.kernel = get_act(attn_kernel)
                attn_type = 'Linear'
            self.attn_norm = get_norm(attn_norm, inner_dim, w_init=0.)
            self.scale_v = Scale(inner_dim, shape=(1, inner_dim, 1))
        self.attn_type = attn_type
        
        # ablation: use_gate or not
        # ablation: which gate_act to use
        gate_act = kwargs.get('gate_act', 'sigmoid')
        gate_act = get_act(gate_act)
        self.no_gate = kwargs.get('no_gate', False)
        if not self.no_gate:  
            self.g = ConvGate(attn_dim, inner_dim, act=gate_act, conv1d=conv_1d)
        self.out = ConvNorm(inner_dim, dim, bn_w_init=0.)
    
    def extra_repr(self):
        return self.attn_type
    
    def softmax_attn(self, q, k, v):
        q = q * self.scale
        attn = q.transpose(-2, -1) @ k
        return v @ attn.softmax(-1).transpose(-2, -1)
    
    def linear_attn(self, q, k, v):
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
        x = self.dwc(self.proj(x))  # local aggregation
        if self.no_attn:
            if not self.no_gate:
                x = x * self.g(x)
            return self.out(x)
        else:
            x = x.flatten(2)
            q, k, v = x.split(self.slices, dim=1)
            attn = self.attn(q, k, v)
            if not self.no_gate:
                attn = attn * self.g(x)  # input-dependent gating
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


class FocusedLinearAttention(nn.Module):
    def __init__(self, dim, attn_ratio, attn_norm, dwc_kernel, **kwargs):
        super().__init__()
        
        d = int(dim * attn_ratio)
        self.dim = dim
        self.qkv = ConvNorm(dim, d * 3)
        self.kernel = nn.ReLU()
        self.dwc = RepConv(d, d, k=dwc_kernel, g=d)
        self.eps = 1e-5
        self.attn_norm = get_norm(attn_norm, d, w_init=0.)
        self.proj = ConvNorm(d, dim, bn_w_init=0.)

    def _attn(self, q, k, v):
        kT = k.transpose(-1,-2)
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
        B, _ ,H, W = x.shape
        q, k, v = self.qkv(x).flatten(2).chunk(3, dim=1)
        q = self.kernel(q) + self.eps
        k = self.kernel(k) + self.eps
        q = F.normalize(q.square())
        k = F.normalize(k.square())
        
        attn = self.attn_norm(self._attn(q,k,v)).reshape(B, -1, H, W)
        x = attn + self.dwc(v.reshape(B, -1, H, W))
        return self.proj(x)
    

class EfficientViT(nn.Module):
    def __init__(self, dim, attn_ratio, attn_norm, **kwargs):
        super().__init__()
        
        d = int(dim * attn_ratio)
        attn_dim = 3 * d
        scales=(5,)
        
        self.dim = dim
        self.qkv = ConvNorm(dim, attn_dim)
        self.aggreg = nn.ModuleList([
            nn.Sequential(
                ConvNorm(attn_dim, attn_dim, k=k, p=(k // 2), g=attn_dim),
                ConvNorm(attn_dim, attn_dim, k=1, g=attn_dim),
            ) for k in scales
        ])
        self.kernel = nn.ReLU()
        self.eps = 1e-5
        
        inner_dim = d * (1 + len(scales))
        self.attn_norm = get_norm(attn_norm, inner_dim, w_init=0.)
        self.proj = ConvNorm(inner_dim, dim, bn_w_init=0.)
    
    def _attn(self, q, k, v):
        kT = k.transpose(-1,-2)
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

        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        qkv = torch.cat(multi_scale_qkv, dim=1)  # B, C', H, W
        q, k, v = qkv.flatten(2).chunk(3, dim=1)  # B, D, N

        # lightweight global attention
        q = self.kernel(q) + self.eps
        k = self.kernel(k) + self.eps
        x = self.attn_norm(self._attn(q, k, v)) + v
        return self.proj(x.reshape(B, -1, H, W))
        

class VanillaLinearAttn(nn.Module):
    def __init__(self, dim, attn_ratio, attn_norm, **kwargs):
        super().__init__()
        
        d = int(dim * attn_ratio)
        
        self.dim = dim
        self.qkv = ConvNorm(dim, 3 * d)
        self.kernel = get_act('elu1')
        self.eps = 1e-5
        
        self.attn_norm = get_norm(attn_norm, d, w_init=0.)
        self.proj = ConvNorm(d, dim, bn_w_init=0.)
    
    def _attn(self, q, k, v):
        kT = k.transpose(-1,-2)
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
        q, k, v = self.qkv(x).flatten(2).chunk(3, dim=1)  # B, D, N
        q, k = self.kernel(q), self.kernel(k)
        x = self.attn_norm(self._attn(q, k, v)) + v
        return self.proj(x.reshape(B, -1, H, W))
        
        
class ELABlock(nn.Module):
    def __init__(self, dim, attn_ratio, attn_kernel, attn_norm, 
                 dwc_kernel, mlp_ratio, act, drop_path, **kwargs):
        super().__init__()
        # ablation: use residual scale or not
        # ablation: use LayerNorm or not
        no_rs = kwargs.get('no_rs', False)
        use_ln = kwargs.get('use_ln', False)
        norm = 'ln2d' if use_ln else 'identity'
        
        self.norm1 = get_norm(norm, dim)
        self.ela = ELA(dim, attn_ratio, attn_kernel, attn_norm, dwc_kernel, **kwargs)
        self.dp1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.rs1 = Scale(dim) if not no_rs else nn.Identity()
        
        # ablation: which linear attention to use
        linear_type = kwargs.get('linear_type', 'ela')
        if linear_type != 'ela':
            if linear_type == 'flatten':
                self.ela = FocusedLinearAttention(dim, attn_ratio, attn_norm, dwc_kernel)
            elif linear_type == 'effvit':
                self.ela = EfficientViT(dim, attn_ratio, attn_norm)
            elif linear_type == 'vanilla':
                self.ela = VanillaLinearAttn(dim, attn_ratio, attn_norm)
            else:
                raise NotImplementedError
        
        self.norm2 = get_norm(norm, dim)
        self.mlp = MLP(dim, mlp_ratio, act)
        self.dp2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.rs2 = Scale(dim) if not no_rs else nn.Identity()

    def forward(self, x):
        x = self.rs1(x) + self.dp1(self.ela(self.norm1(x)))
        return self.rs2(x) + self.dp2(self.mlp(self.norm2(x)))


class MBConv(nn.Module):
    def __init__(self, inp, oup, exp, k, s, act, fuse=False, **kwargs):
        super().__init__()
        hid = int(oup * exp)
        res = (s == 1 and inp == oup)
        
        no_rep = kwargs.get('no_rep', False)
        
        if fuse:
            ops = [RepConv(inp, hid, k, s, res=res, use_rep=not no_rep)]  # c,h,w -> hid,h/2,w/2
        else:
            ops = [
                ConvNorm(inp, hid), get_act(act),  # c -> hid
                RepConv(hid, hid, k, s, g=hid, res=res, use_rep=not no_rep) # h,w -> h/2,w/2
            ]
        ops += [get_act(act), ConvNorm(hid, oup)]  # hid -> 2c 
        self.conv = nn.Sequential(*ops)

    def forward(self, x): 
        return self.conv(x) # b,c,h,w -> b,2c,h/2,w/2
    

class BasicLayer(nn.Module):
    def __init__(self, inp, oup, depth, ds_exp, ds_kernel, ds_fuse, act, dwc_kernel, block_type, agg_ratio, 
                 attn_ratio, attn_kernel, attn_norm, mlp_ratio, dp_rates, use_checkpoint, **kwargs
                 ):
        super().__init__()
        # downsample
        blocks = [MBConv(inp=inp, oup=oup, exp=ds_exp, k=ds_kernel, s=2, act=act, fuse=ds_fuse, **kwargs)]
        
        # blocks
        assert block_type in ['VLA', 'GAB'], f'block_type must be VLA or GAB, but got {block_type}'
        if block_type == 'GAB':
            blocks += [GatedAggBlock(dim=oup, k=dwc_kernel, act=act, drop_path=dp_rates[i], e=agg_ratio, **kwargs) for i in range(depth)]
        else:
            blocks += [ELABlock(
                dim=oup, attn_ratio=attn_ratio, attn_kernel=attn_kernel, attn_norm=attn_norm,
                dwc_kernel=dwc_kernel, mlp_ratio=mlp_ratio, act=act, drop_path=dp_rates[i], **kwargs) for i in range(depth)]
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

class LiteVLA_AB(nn.Module):
    def __init__(self, inp=3, num_classes=1000, dims=(48, 96, 192, 384), 
                 depths=(2, 2, 11, 3), block_types=('GAB', 'GAB', 'VLA', 'VLA'), 
                 stem_kernel=5, ds_exp=4, ds_kernel=3, ds_fuse="ff--", act='silu', dwc_kernel=5, agg_ratio=0.5,
                 attn_ratio=1/8, attn_kernel='elu1', attn_norm='mrms', mlp_ratio=3, head_norm='mrms',
                 drop_path_rate=0.,
                 use_checkpoint=False, distillation=False,  # for training
                 backbone=False, out_indices=None, pretrained=None,  # for backbone
                 **kwargs):
        super().__init__()
        if isinstance(dims, int):
            dims = [dims * 2 ** i for i in range(len(depths))]
        ds_fuse = [f == 'f' for f in ds_fuse]
    
        ## ablation: use LayerNorm or not
        if kwargs.get('use_ln'):
            attn_norm = 'identity'
            head_norm = 'ln'
        
        # Stem
        dims = [dims[0] // 2, *dims]
        no_rep = kwargs.get('no_rep', False)
        self.stem = nn.Sequential(RepConv(inp, dims[0], k=stem_kernel, s=2, use_rep=not no_rep), get_act(act))
        
        # build layers [ DownSample + Blocks ] x 4
        dpr_lst = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList([BasicLayer(
            inp=dims[i], oup=dims[i+1], depth=depths[i], ds_exp=ds_exp, ds_kernel=ds_kernel, 
            ds_fuse=ds_fuse[i], act=act, dwc_kernel=dwc_kernel, block_type=block_types[i], agg_ratio=agg_ratio,
            attn_ratio=attn_ratio, attn_kernel=attn_kernel, attn_norm=attn_norm,
            mlp_ratio=mlp_ratio, dp_rates=dpr_lst[sum(depths[:i]):sum(depths[:i+1])], 
            use_checkpoint=use_checkpoint, 
            **kwargs
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
    
    # TODO [HSC]: check if this could improve performance
    #         self.train()
    
    # def train(self, mode=True):
    #     """Convert the model into training mode while keep layers freezed."""
    #     super(LiteVLA_AB, self).train(mode)
    #     if mode:
    #         for m in self.modules():
    #             if isinstance(m, _BatchNorm):
    #                 m.eval()

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



def get_ablation_args(args):
    args = args.get('ablation', "")
    args = [a for a in args.split('-') if len(a) > 0]
    
    updates = dict()
    for arg in args:
        lst = arg.split(':')
        assert len(lst) == 2 or len(lst) == 1
        arg = lst[0]        

        
        ### residual connection in block
        if arg == 'no_rs':
            updates['no_rs'] = True
            
            
        ### efficient linear attn
        if arg == 'no_attn':
            updates['no_attn'] = True

        if arg == 'use_sa':
            updates['use_sa'] = True
        
        if arg == 'linear_type':
            updates['linear_type'] = lst[1]
        
        
        ### input-dependent gating
        if arg == 'gate_act':
            updates['gate_act'] = lst[1]
            
        if arg == 'no_gate':
            updates['no_gate'] = True
        
        
        ### norm
        if arg == 'no_attn_norm':
            updates['attn_norm'] = 'identity'
        
        if arg == 'use_ln':
            updates['use_ln'] = True
            
            
        ### repconv
        if arg == 'no_rep':
            updates['no_rep'] = True
            

    return updates


class LiteVLA(LiteVLA_AB):
    def __init__(self, *args, **kwargs):
        updates = get_ablation_args(kwargs)
        kwargs.update(updates)
        kwargs.pop('ablation', None)
        super().__init__(*args, **kwargs)