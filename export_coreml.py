import torch

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

# show_arch, show_table = True, True
show_arch, show_table = False, False

for out_idx, img_size in versions:
    print(f'out_idx: {out_idx}, img_size: {img_size}')
    
    # version = 'litevla_t'
    # model = create_model(version, out_indices=out_idx)
    
    # for ela
    cfg = 'configs/ablation/for_ela/litevla_n_all_la.yaml'
    cfg = 'configs/ablation/for_ela/litevla_n_no_attn.yaml'
    cfg = 'configs/ablation/for_ela/litevla_n_use_sa.yaml'
    
    # for gate
    cfg = 'configs/ablation/for_gate/litevla_n_no_gate.yaml'
    
    # for rep
    cfg = 'configs/ablation/for_rep/litevla_n_no_rep.yaml'
    
    
    model = build_model(get_config(Args(cfg=cfg)))
    version = cfg.replace('.yaml', '').split('/')[-1]
    
    rep_model = reparameterize_model(model)
    
    _ = model(torch.rand(2, 3, *img_size))
    _ = rep_model(torch.rand(2, 3, *img_size))

    fvcore_flop_count(model=model, input_shape=(3, *img_size), show_arch=show_arch, show_table=show_table)
    fvcore_flop_count(model=rep_model, input_shape=(3, *img_size), show_arch=show_arch, show_table=show_table)

    model = rep_model
    model.eval()

    example_input = torch.rand(1, 3, *img_size)
    traced_model = torch.jit.trace(model, example_input)
    out = traced_model(example_input)
    
    model_name = f'coreml_{version}'
    model = ct.convert( traced_model, inputs=[ct.ImageType(shape=example_input.shape)] )
    model.save(f"coreml/{model_name}_{img_size[0]}.mlpackage")
    print(f"Saved coreml/{model_name}_{img_size[0]}.mlpackage")
    print()
print()