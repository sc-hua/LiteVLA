from .litevla import LiteVLA, create_litevla
from .litevla_for_ab import LiteVLA as LiteVLA_AB
    
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
            # drop_path_ratio=config.MODEL.DROP_PATH_RATE,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT or kwargs.pop('use_checkpoint', False),
            distillation=config.MODEL.LITEVLA.DISTILLATION or kwargs.pop('distillation', False),
            backbone=config.MODEL.LITEVLA.BACKBONE or kwargs.pop('backbone', False),
            out_indices=config.MODEL.LITEVLA.OUT_INDICES,
            pretrained=config.MODEL.LITEVLA.PRETRAINED or kwargs.pop('pretrained', None),
            ablation=config.MODEL.LITEVLA.ABLATION,
            **kwargs)
    return model


import timm

def build_timm_model(config, is_pretrain=False, **kwargs):
    model = None
    model_type = config.MODEL.TYPE
    if model_type.startswith('timm'):
        for prefix in ['timm_', 'timm.', 'timm/', 'timm-', 'timm+']:
            model_type = model_type.replace(prefix, '')
        print(f"Loading timm model: {model_type}")
        model = timm.create_model(model_type, pretrained=is_pretrain, **kwargs)
        use_checkpoint = config.TRAIN.USE_CHECKPOINT or kwargs.pop('use_checkpoint', False)
        model.set_grad_checkpointing(enable=use_checkpoint)
    return model

def build_model(config, **kwargs):
    model = build_timm_model(config, **kwargs)
    if model is None:
        model = build_litevla_model(config, **kwargs)
    return model

