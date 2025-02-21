import torch

from models.ops.lib import fvcore_flop_count
from models import create_litevla
from timm.utils.model import reparameterize_model
try:
    import coremltools as ct
except ModuleNotFoundError:
    print('coremltools is not installed.')
    

versions = [
    (None, (224, 224)),
    # ((0,1,2,3), (512, 512)),
]

# show_arch, show_table = True, True
show_arch, show_table = False, False

for out_idx, img_size in versions:
    print(f'out_idx: {out_idx}, img_size: {img_size}')
    version = 't'
    model = create_litevla(version, out_indices=out_idx)
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
    
    model_name = f'coreml_litevla_{version}'
    model = ct.convert( traced_model, inputs=[ct.ImageType(shape=example_input.shape)] )
    model.save(f"coreml/{model_name}_{img_size[0]}.mlpackage")
    print(f"Saved coreml/{model_name}_{img_size[0]}.mlpackage")
    print()
print()