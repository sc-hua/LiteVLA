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


# for flop and parameter count
def fvcore_flop_count(model: torch.nn.Module, inputs=None, input_shape=(3, 224, 224), 
                      show_table=False, show_arch=False, verbose=True):
    from fvcore.nn.parameter_count import parameter_count as fvcore_parameter_count
    from fvcore.nn.flop_count import flop_count, FlopCountAnalysis
    from fvcore.nn.print_model_statistics import flop_count_str, flop_count_table

    if inputs is None:
        assert input_shape is not None
        if len(input_shape) == 1:
            input_shape = (1, 3, input_shape[0], input_shape[0])
        elif len(input_shape) == 2:
            input_shape = (1, 3, *input_shape)
        elif len(input_shape) == 3:
            input_shape = (1, *input_shape)
        else:
            assert len(input_shape) == 4

        inputs = (torch.randn(input_shape).to(next(model.parameters()).device),)

    model.eval()

    Gflops, unsupported = flop_count(model=model, inputs=inputs)

    flops_table = flop_count_table(
        flops=FlopCountAnalysis(model, inputs),
        max_depth=100,
        activations=None,
        show_param_shapes=True,
    )

    flops_str = flop_count_str(flops=FlopCountAnalysis(model, inputs), activations=None)

    if show_arch:
        print(flops_str)

    if show_table:
        print(flops_table)

    params = fvcore_parameter_count(model)[""]
    flops = sum(Gflops.values())

    if verbose:
        print(Gflops.items())
        print("GFlops: ", flops, "Params: ", params, flush=True)

    return params, flops


# remove prefix
def remove_prefix(name, prefix):
    for p in [f"{prefix}{x}" for x in ['_', '.', '/', '-', '+']]:
        name = name.replace(p, '')
    return name


