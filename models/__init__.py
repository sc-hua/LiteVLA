import timm
from .litevla import LiteVLA
from .litevla_for_ab import LiteVLA as LiteVLA_AB
from .lib import remove_prefix


""" 
    Build model from config and kwargs 
"""
def build_model(config, **kwargs):
    model = build_timm_model(config, **kwargs)
    if model is None:
        model = build_litevla_model(config, **kwargs)
    return model

def build_timm_model(config, is_pretrain=False, **kwargs):
    model = None
    model_type = config.MODEL.TYPE
    model_type = model_type.lower()
    if model_type.startswith('timm'):
        model_type = remove_prefix(model_type, 'timm')
        print(f"Loading timm model: {model_type}")
        model = timm.create_model(model_type, pretrained=is_pretrain, **kwargs)
        use_checkpoint = config.TRAIN.USE_CHECKPOINT or kwargs.pop('use_checkpoint', False)
        model.set_grad_checkpointing(enable=use_checkpoint)
    return model
    
def build_litevla_model(config, **kwargs):
    model = None
    if 'litevla' in config.MODEL.TYPE:
        MODEL = LiteVLA
        ablation = config.MODEL.LITEVLA.ABLATION
        if ablation != '' and len(ablation) > 0:
            MODEL = LiteVLA_AB
        model = MODEL(
            dims=config.MODEL.LITEVLA.DIMS,
            depths=config.MODEL.LITEVLA.DEPTHS,
            block_types=config.MODEL.LITEVLA.BLOCK_TYPES,
            stem_kernel=config.MODEL.LITEVLA.STEM_KERNEL,
            ds_exp=config.MODEL.LITEVLA.DS_EXP,
            ds_kernel=config.MODEL.LITEVLA.DS_KERNEL,
            ds_fuse=config.MODEL.LITEVLA.DS_FUSE,
            act=config.MODEL.LITEVLA.ACT,
            dwc_kernel=config.MODEL.LITEVLA.DWC_KERNEL,
            attn_ratio=config.MODEL.LITEVLA.ATTN_RATIO,
            attn_kernel=config.MODEL.LITEVLA.ATTN_KERNEL,
            attn_norm=config.MODEL.LITEVLA.ATTN_NORM,
            mlp_ratio=config.MODEL.LITEVLA.MLP_RATIO,
            head_norm=config.MODEL.LITEVLA.HEAD_NORM,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT or kwargs.pop('use_checkpoint', False),
            distillation=config.MODEL.LITEVLA.DISTILLATION or kwargs.pop('distillation', False),
            backbone=config.MODEL.LITEVLA.BACKBONE or kwargs.pop('backbone', False),
            out_indices=config.MODEL.LITEVLA.OUT_INDICES,
            pretrained=config.MODEL.LITEVLA.PRETRAINED or kwargs.pop('pretrained', None),
            ablation=config.MODEL.LITEVLA.ABLATION,
            **kwargs)
    return model



""" 
    Directly get model from name and kwargs 
"""
def create_model(name, **kwargs):
    name = name.lower()
    if name.startswith('timm'):
        return create_timm_model(name, **kwargs)
    elif name.startswith('litevla'):
        return create_litevla(name, **kwargs)
    else:
        raise NotImplementedError(f'Unknown model: {name}')
    
    
def create_timm_model(name, **kwargs):
    name = remove_prefix(name, 'timm')
    if kwargs.get('out_indices', None) is not None:
        kwargs['features_only'] = True
    model = timm.create_model(name, **kwargs)
    return model
    
    
default_args = dict(
    inp=3, num_classes=1000, dims=(48, 96, 192, 384), 
    depths=(2, 2, 11, 3), block_types=('GAB', 'GAB', 'VLA', 'VLA'), 
    stem_kernel=5, ds_exp=4, ds_kernel=3, ds_fuse="ff--", act='silu', dwc_kernel=5,
    attn_ratio=1/8, attn_kernel='elu1', attn_norm='mrms', mlp_ratio=3, head_norm='mrms',
    use_checkpoint=False, distillation=False,  # for training
    backbone=False, out_indices=None, pretrained=None,  # for backbone
)



def get_litevla_version_args(version):
    version_args = None
    if version in ['tiny', 't']:
        version_args = dict(dims=(32, 64, 128, 256), depths=(2, 2, 7, 3),
                      block_types=('GAB', 'GAB', 'GAB', 'VLA'), ds_fuse="ffff")
    if version in ['small', 's']:
        version_args = dict(dims=(40, 80, 160, 320), ds_fuse="fff-")
    if version in ['normal', 'n']:
        version_args = dict(dims=(48, 96, 192, 384), ds_fuse="ff--")
    if version in ['medium', 'm']:
        # version_args = dict(dims=(56, 112, 224, 448), depths=(2, 2, 11, 3), ds_fuse="f---")
        version_args = dict(dims=(56, 112, 224, 448), depths=(2, 2, 10, 4), 
                            ds_fuse="f---", drop_path_rate=0.1)
    if version in ['large', 'l']:
        # version_args = dict(dims=(64, 128, 256, 512), ds_fuse="f---")
        version_args = dict(dims=(64, 128, 256, 512), 
                            ds_fuse="f---", drop_path_rate=0.2)
    if version_args is None:
        raise NotImplementedError(f"LiteVLA: version({version}) not implemented")
    return version_args



"""
    for image classification on imagenet-1k
"""
def create_litevla(name, **kwargs):
    name = remove_prefix(name, 'litevla')
    
    MODEL = LiteVLA
    ablation = kwargs.pop('ablation', '')
    if ablation != '' and len(ablation) > 0:
        MODEL = LiteVLA_AB
    
    return MODEL(**{
        **get_litevla_version_args(name), 
        **kwargs, # [caution]: kwargs will override version_args
    })



""" 
    for downstream: Object Detection / Semantic Segmentation
"""
has_mmdet = True
has_mmseg = True

try:
    from mmengine.model import BaseModule
    from mmdet.registry import MODELS as MODELS_MMDET
except ImportError:
    # print('\t\t>> If for Object Detection, please install mmdetection first.')
    has_mmdet = False

try: 
    from mmengine.model import BaseModule
    from mmseg.registry import MODELS as MODELS_MMSEG
except ImportError:
    # print('\t\t>> If for Semantic Segmentation, please install mmsegmentation first.')
    has_mmseg = False
    

if has_mmdet and has_mmseg:
    @MODELS_MMSEG.register_module()
    @MODELS_MMDET.register_module()
    class MM_LITEVLA(BaseModule, LiteVLA):
        def __init__(self, *args, **kwargs):
            BaseModule.__init__(self)
            
            # specify model version
            version = kwargs.pop('version', None)
            if version is None:
                version = 'litevla_n'
                print('\t\t>> LiteVLA: version is not specified, use default: LiteVLA-N')
            version_args = get_litevla_version_args(remove_prefix(version, 'litevla'))
            # [caution]: kwargs will override version_args
            kwargs = {**version_args, **kwargs}
            
            LiteVLA.__init__(self, *args, **kwargs)
