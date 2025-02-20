import torch
from copy import deepcopy

GLOBAL_EPS = 5e-4  # fp16: 2^(-14) ~ 65504

# for reparameterization trick
def fuse_conv_bn(kernel, bn):
    weight = kernel * (bn.weight / (bn.running_var + bn.eps).sqrt()).reshape(-1, 1, 1, 1)
    bias = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps).sqrt()
    return weight, bias


# for reparameterization trick
def get_id_tensor(c):
    oup = c.out_channels
    g_d, k1, k2 = c.weight.shape[1:]
    id_weight = torch.zeros(
        (oup, g_d, k1, k2), 
        device=c.weight.device, 
        dtype=c.weight.dtype
    )
    for i in range(oup):
        id_weight[i, i % g_d, k1 // 2, k2 // 2] = 1
    return id_weight
    

# for deleting attributes
def delete_attr(module, attr_lst):
    assert isinstance(attr_lst, (str, list, tuple)), 'attr_lst must be str or list or tuple'
    attr_lst = [attr_lst, ] if isinstance(attr_lst, str) else attr_lst
    for a in attr_lst:
        if hasattr(module, a):
            delattr(module, a)


# for linear attention
def use_linear(q, v):
    Dkq, Dv = q.shape[-2], v.shape[-2]
    Nvk, Nq = v.shape[-1], q.shape[-1]
    return Nvk * Nq * (Dkq + Dv) > (Nvk + Nq) * Dkq * Dv


# for reparameterization trick
def reparameterize_model(model: torch.nn.Module, inplace=False) -> torch.nn.Module:
    """ 
    Same with `timm.utils.model.reparameterize_model`, can be called as line below
    > from timm.utils.model import reparameterize_model 
    """
    if not inplace:
        model = deepcopy(model)

    def _fuse(m):
        for child_name, child in m.named_children():
            if hasattr(child, 'fuse'):
                setattr(m, child_name, child.fuse())
            elif hasattr(child, "reparameterize"):
                child.reparameterize()
            elif hasattr(child, "switch_to_deploy"):
                child.switch_to_deploy()
            _fuse(child)

    _fuse(model)
    return model