import torch
import torch.nn as nn

from models.lib import fvcore_flop_count
from models import create_model
from timm.utils.model import reparameterize_model
try:
    import coremltools as ct
except ModuleNotFoundError:
    print('coremltools is not installed.')

from models import build_model
from analyze.my_args import Args
from config import get_config
from models import build_model

versions = [
    (None, (224, 224)),
    # ((0,1,2,3), (512, 512)),
]

def replace_forward(model, replace_lst=[], replact_to=None):
    assert replact_to is not None
    assert len(replace_lst) > 0
    def _replace_forward(m):
        for child_name, child in m.named_children():
            for replace in replace_lst:
                if replace in child_name:
                    setattr(m, child_name, replact_to())
            _replace_forward(child)
    _replace_forward(model)
    return model

# show_arch, show_table = True, True
show_arch, show_table = False, False

for out_idx, img_size in versions:
    
    """" load from version """
    # version = 'litevla_m'
    # model = create_model(version, out_indices=out_idx)
    # model_name = f'coreml_{version}'
    # version = 'iformer_l_faster'
    # version = 'iformer_l'
    # from others import get_model
    # # model = get_model(version, out_indices=[0,1,2,3])
    # model = get_model(version)
    # model_name = f'coreml_{version}'
    
    
    """ load from config yaml """
    # # for block
    # # # cfg = 'configs/ablation/for_block/litevla_n_0gab4mla.yaml'  # all la
    # cfg = 'configs/ablation/for_block/litevla_n_1gab3mla.yaml'
    # # # cfg = 'configs/ablation/for_block/litevla_n_2gab2mla.yaml'   # final
    # cfg = 'configs/ablation/for_block/litevla_n_3gab1mla.yaml'
    # cfg = 'configs/ablation/for_block/litevla_n_4gab0mla.yaml'
    # cfg = 'configs/ablation/for_block/litevla_n_use_ln.yaml'
    
    # # for ela
    # cfg = 'configs/ablation/for_ela/litevla_n_all_la.yaml'
    # cfg = 'configs/ablation/for_ela/litevla_n_attn_ratio_5.yaml'
    # cfg = 'configs/ablation/for_ela/litevla_n_attn_ratio_25.yaml'
    # cfg = 'configs/ablation/for_ela/litevla_n_no_attn.yaml'
    # cfg = 'configs/ablation/for_ela/litevla_n_use_sa.yaml'
    # cfg = "configs/ablation/for_ela/litevla_n_effvit.yaml"
    # cfg = "configs/ablation/for_ela/litevla_n_flatten.yaml"
    # cfg = "configs/ablation/for_ela/litevla_n_vanilla.yaml"
    
    # # # for gate
    # cfg = 'configs/ablation/for_gate/litevla_n_no_gate.yaml'
    # cfg = 'configs/ablation/for_gate/litevla_n_gate_act_silu.yaml'
    
    # # for norm
    # cfg = 'configs/ablation/for_norm/litevla_n_no_attn_norm.yaml'
    # cfg = 'configs/ablation/for_norm/litevla_n_rms_attn_norm.yaml'
    
    # # # for rep
    # # cfg = 'configs/ablation/for_rep/litevla_n_no_rep.yaml'
    
    # # for scale
    # cfg = "configs/ablation/for_scale/litevla_m_v1.yaml"
    cfg = "configs/ablation/for_scale/litevla_m_v2.yaml"
    
    model = build_model(get_config(Args(cfg=cfg)))
    version = cfg.replace('.yaml', '').split('/')[-1]
    version = 'ab_' + version
    model_name = f'coreml_{version}'
    
    
    """ load from timm """
    # import timm
    # import others
    # # img_size = (300, 300)
    # name = "swin_tiny_patch4_window7_224"
    # model = timm.create_model(
    #     name, 
    #     features_only=(out_idx is not None), 
    #     # fork_feat=(out_idx is not None),
    #     # intermediates_only=(out_idx is not None),
    #     img_size=img_size[0],
    # )
    model = replace_forward(model, replace_lst=['drop_path', 'dp'], replact_to=nn.Identity)
    # model_name = f'timm_{name}'
    """ end """
    
    if out_idx is not None:
        model_name = f'backbone_{model_name}'
    
    model.eval()
    rep_model = reparameterize_model(model)
    
    _ = model(torch.rand(2, 3, *img_size))
    _ = rep_model(torch.rand(2, 3, *img_size))

    print(f'out_idx: {out_idx}, img_size: {img_size}')
    fvcore_flop_count(model=model, input_shape=(3, *img_size), show_arch=show_arch, show_table=show_table)
    fvcore_flop_count(model=rep_model, input_shape=(3, *img_size), show_arch=show_arch, show_table=show_table)

    model = rep_model
    example_input = torch.rand(1, 3, *img_size)
    traced_model = torch.jit.trace(model, example_input)
    out = traced_model(example_input)
    
    model = ct.convert( traced_model, inputs=[ct.ImageType(shape=example_input.shape)] )
    model.save(f"coreml/{model_name}_{img_size[0]}.mlpackage")
    print(f"Saved coreml/{model_name}_{img_size[0]}.mlpackage")
    print()
print()