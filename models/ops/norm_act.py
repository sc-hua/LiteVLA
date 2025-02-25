import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.norm import GroupNorm1, LayerNorm2d

def get_act(name):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    if name == 'gelu':
        return nn.GELU()
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'silu':
        return nn.SiLU()
    if name == 'elu1':
        return ELU_1()
    if name in ['id', 'identity']:
        return nn.Identity()
    raise NotImplementedError(f'Unknown act: {name}')


class ELU_1(nn.Module):
    """ for linear attention """
    def forward(self, x):
        return F.elu(x) + 1



def get_norm(name, dim, **kwargs):
    name = name.lower()
    if name in ['id', 'identity']:
        return nn.Identity()
    
    w_init = kwargs.pop('w_init', 1.)  # initial value for weight
    b_init = kwargs.pop('b_init', 0.)  # initial value for bias
    norm = None
    
    # BatchNorm
    if name == 'bn':
        norm = nn.BatchNorm1d(dim, **kwargs)
    if name == 'bn2d':
        norm = nn.BatchNorm2d(dim, **kwargs)
    
    # LayerNorm
    if name == 'ln':
        norm = nn.LayerNorm(dim, **kwargs)
    if name == 'ln2d':  # implemented by timm
        norm = LayerNorm2d(dim, **kwargs)
    
    # GroupNorm
    if name == 'gn':
        norm = nn.GroupNorm(dim, **kwargs)
    if name == 'gn1' or name == 'mln':
        # mln is for `modified layer norm`, from metaformer
        # can be implemented by setting group=1 in GroupNorm
        norm = GroupNorm1(dim, **kwargs)
        
    # Ours
    if name == 'mrms':
        norm = ModifiedRMSNorm(dim, w_init=w_init, **kwargs)
    
    # init the values of weight and bias
    if norm is not None:
        if hasattr(norm, 'weight'):
            nn.init.constant_(norm.weight, w_init)
        if hasattr(norm, 'bias'):
            nn.init.constant_(norm.bias, b_init)
        return norm
    else:
        raise NotImplementedError(f'Unknown norm: {name}')


class ModifiedRMSNorm(nn.Module):
    r""" Modified Root Mean Square Normalization.
    The only difference with RMSNorm is that MRMSNorm is taken over all dimensions 
    except the batch dimension.
    
    Modified RMSNorm:
    y = x / MRMS(x) * gamma, where MRMS(x) = sqrt( sum_{i=1}^{n}(x^2) / n + eps )

    Args:
        dim (int): number of channels
        eps (float): small number to avoid division by zero, default: 1e-5
        w_init (float): initial value for weight, default: 1.
        affine (boolean): whether to use affine transformation, default: True
        
    Shape:
        - Input / Output: (B, C, *)
    """
    def __init__(self, dim, eps=1e-5, w_init=1., affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.w_init = w_init
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(w_init * torch.ones(dim))

    def extra_repr(self):
        return "{dim}, eps={eps}, affine={affine}, weight_init={w_init}".format(**self.__dict__)
    
    def get_dims_shape(self, x):
        dims = tuple(range(1, x.dim()))
        shape = [-1 if i == 1 else 1 for i in range(x.dim())]
        return dims, shape
    
    def _norm(self, x, dims):
        return x * torch.rsqrt(x.pow(2).mean(dims,True) + self.eps)

    def forward(self, x):
        dims, shape = self.get_dims_shape(x)
        normlized_x = self._norm(x.float(), dims=dims).to(x.dtype)
        if self.affine:
            return normlized_x * self.weight.view(*shape)
        else:
            return normlized_x