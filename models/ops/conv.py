import torch
import torch.nn as nn

from functools import partial
from timm.layers import trunc_normal_, to_2tuple
from models.lib import GLOBAL_EPS, fuse_conv_bn, get_id_tensor


class Scale(nn.Module):
    """Learnable scale for specified dimension(s).

    Args:
        dim (int): Number of channels.
        init_value (float): Initial value of scale.
        shape (tuple or None): Shape of scale vector for element-wise multiplication.
            Default None is (1, dim, 1, 1), suitable for (B, C, H, W).
    """
    def __init__(self, dim, init_value=1., shape=None):
        super().__init__()
        self.shape = (1, dim, 1, 1) if shape is None else shape
        self.alpha = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.alpha.view(*self.shape)


class ConvNorm(nn.Sequential):
    """ Convolution followed by normalization.

    Args:
        inp (int): input channels
        oup (int): output channels
        k (int or tuple): kernel size
        s (int or tuple): stride
        p (int or None): padding (auto if None)
        d (int): dilation
        g (int): groups
        bn_w_init (float): batch norm weight init (default 1.)
            - Suggestion: use small value when this module directly add residual
            - kinda like `layer_scale` in some models
    """
    def __init__(self, inp, oup, k=1, s=1, p=None, d=1, g=1, bn_w_init=1.):
        super().__init__()
        p = (k // 2) if p is None else p  # auto padding
        self.conv_args = (inp, oup, k, s, p, d, g)
        self.add_module('c', nn.Conv2d(*self.conv_args, bias=False))
        self.add_module('bn', nn.BatchNorm2d(oup))
        nn.init.constant_(self.bn.weight, bn_w_init)
        nn.init.constant_(self.bn.bias, 0.)

    @torch.no_grad()
    def fuse(self):
        w, b = fuse_conv_bn(self.c.weight, self.bn)
        m = nn.Conv2d(*self.conv_args, device=w.device, dtype=w.dtype)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class RepConv(nn.Module):
    r""" Re-parameterized Convolution
    
    Args:
        inp, oup (int): number of input / output channels
        k, s, g (int): kernel size, stride, groups
        use_rep (bool): whether to use reparameterization (mutil-kernel conv), Default: True
        res (bool): whether to use skip connection, Default: False
        bn_w_init (float): weight initialization, Default: 1.
            Suggestion: use 0. when this module directly add residual
    """
    def __init__(self, inp, oup, k=1, s=1, g=1, use_rep=True, res=False, bn_w_init=1.):
        super().__init__()
        self.kernel = to_2tuple(k)
        self.res = res
        if self.res:
            assert inp == oup and s == 1, \
                f"make sure inp({inp}) == oup({oup}) and stride({s}) == 1 when using skip connection"
            self.scale_res = Scale(oup)
            bn_w_init = 0.
        
        # make sure k > 0
        k_lst = [x for x in range(k, -1, -2) if x > 0] if use_rep else [k]
        self.ops = nn.ModuleList([
            ConvNorm(inp, oup, _k, s, (_k // 2), g=g, bn_w_init=bn_w_init) 
            for _k in k_lst
        ])
        self.repr_str = (
            f"# {'RepConv' if g == 1 else 'RepDWConv'}:"
            f" kernels={k_lst}"
            f"{', w. res' if res else ''}"
        )
    
    def extra_repr(self):
        return self.repr_str
    
    def forward(self, x, out=0):
        for op in self.ops:
            out = out + op(x)
        if self.res:
            out = out + self.scale_res(x)
        return out

    @torch.no_grad()
    def fuse(self):
        c = self.ops[0]
        if hasattr(c, 'fuse'):
            c = c.fuse()
        
        lk = self.kernel
        weight, bias = 0, 0
        for op in self.ops:
            if hasattr(op, 'fuse'):
                op = op.fuse()
            w, b, sk = op.weight, op.bias, op.kernel_size
            if sk != lk:
                pad = (lk[0] - sk[0]) // 2, (lk[1] - sk[1]) // 2
                w = nn.functional.pad(w, (pad[1], pad[1], pad[0], pad[0]))
            b = b if b is not None else 0
            weight, bias = weight + w, bias + b
        
        if self.res:
            weight += get_id_tensor(c) * self.scale_res.alpha.view(-1, 1, 1, 1)

        # fuse into one conv
        rep_conv = nn.Conv2d(
            in_channels=c.in_channels, out_channels=c.out_channels, kernel_size=c.kernel_size, 
            stride=c.stride, padding=c.padding, dilation=c.dilation, groups=c.groups, 
            device=weight.device, dtype=weight.dtype
        )
        rep_conv.weight.data.copy_(weight)
        rep_conv.bias.data.copy_(bias)

        # set extra_repr for debug
        if len(self.ops) > 1:
            repr_str = f"{self.repr_str}\n{rep_conv.extra_repr()}"
            rep_conv.extra_repr = partial(lambda m: repr_str, rep_conv)
        return rep_conv


class BNLinear(nn.Sequential):
    r""" Batch Normalization + Linear
    
    Args:
        inp, oup (int): number of input / output channels
        std (float): standard deviation, Default: 0.02
        use_conv2d (bool): whether to use conv2d (kernel=1) instead of linear, Default: False
    """
    def __init__(self, inp, oup, std=0.02, use_conv2d=False):
        super().__init__()
        bn = nn.BatchNorm2d(inp) if use_conv2d else nn.BatchNorm1d(inp)
        l = nn.Conv2d(inp, oup, 1) if use_conv2d else nn.Linear(inp, oup)
        
        trunc_normal_(l.weight, std=std)
        nn.init.constant_(l.bias, 0)
        self.add_module('bn', bn)
        self.add_module('l', l)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        use_conv2d = isinstance(l, nn.Conv2d)
        weight = l.weight[:,:,0,0] if use_conv2d else l.weight
        
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = weight * w[None, :]
        b = (weight @ b[:, None]).view(-1) + l.bias
        inp, oup, device = w.size(1), w.size(0), w.device
        if use_conv2d:
            m = nn.Conv2d(inp, oup, 1, 1, 0, device=device)
            w = w[:,:,None,None]
        else:
            m = nn.Linear(inp, oup, device=device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


def ConvGate(inp, oup, act, conv1d=False):
    r""" Conv(Linear) + Gate
    
    Args:
        inp, oup (int): number of input / output channels
        act (None or nn.Module): activation function
        conv1d (bool): whether to use conv1d instead of conv2d, Default: False
    """
    Conv = nn.Conv1d if conv1d else nn.Conv2d
    c = Conv(inp, oup, kernel_size=1, bias=True)
    # override init of c, set weight near zero, bias=1
    # (1.make this `gate branch` more stable; 2.reduces the impact to another branch) for early epochs
    trunc_normal_(c.weight, std=GLOBAL_EPS)
    nn.init.ones_(c.bias)
    if act is None:
        return c
    
    # !!! NOTE: act should be an instance, not a class !!!
    assert isinstance(act, nn.Module), f"Expected `act({act})` to be an `nn.Module` instance."
    return nn.Sequential(c, act)

