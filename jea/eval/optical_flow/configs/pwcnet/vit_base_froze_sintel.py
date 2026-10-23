

_base_ = [
    '../datasets/sintel_pwcnet_384x768.py',
    '../default_runtime.py',
]
custom_imports = dict(
    imports=['jea.eval.optical_flow.register_backbone'],
    allow_failed_imports=False
)

find_unused_parameters=True

model = dict(
    type='PWCNet',
    encoder=dict(
        type='JeaVisionTransformerForPWC',
        img_size=[384, 768],
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_layer='swiglufused',
        init_values=0.1,
        block_chunks=4,
        out_with_norm=True,
        mlp_ratio=4.,
        qkv_bias=True,
        with_fpn=True,
        drop_path_rate=0,
        frozen_stages=4,
        out_indices=(0, 1, 2, 3),
        out_channels=dict(
            level2=32,
            level3=64,
            level4=96,
            level5=128,
        ),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='path_or_url.pth'
        )
    ),
    decoder=dict(
        type='PWCNetDecoder',
        in_channels=dict(
            level5=81,
            level4=181,
            level3=149,
            level2=117,
        ),
        flow_div=20.,
        corr_cfg=dict(type='Correlation', max_displacement=4, padding=0),
        warp_cfg=dict(type='Warp', align_corners=True, use_mask=True),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        scaled=False,
        post_processor=dict(type='ContextNet', in_channels=565),
        flow_loss=dict(
            type='MultiLevelEPE',
            p=1,
            q=0.4,
            eps=0.01,
            reduction='sum',
            weights=dict(
                level2=0.005,
                level3=0.01,
                level4=0.02,
                level5=0.08,
            )),
    ),
    train_cfg=dict(),
    test_cfg=dict(),
    init_cfg=dict(
        type='Kaiming',
        nonlinearity='leaky_relu',
        layer=['Conv2d', 'ConvTranspose2d'],
        mode='fan_in',
        bias=0)
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05)
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12)
evaluation = dict(interval=20)

checkpoint_config = dict(interval=1, max_keep_ckpts=1, save_last=True)

lr_warmup = dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500)
lr_scheduler = dict(type='MultiStepLR', milestones=[8, 11], gamma=0.1, by_epoch=True)

param_scheduler = [lr_warmup, lr_scheduler]

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
