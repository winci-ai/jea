
_base_ = [
    '../datasets/voc2012.py',
    'mmseg::_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['jea.eval.segmentation.register_backbone'],
    allow_failed_imports=False
)
find_unused_parameters=True

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='JeaVisionTransformer',
        img_size=[512],
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        ffn_layer="swiglufused",
        init_values=0.1,
        block_chunks=4,
        out_with_norm=True,
        mlp_ratio=4.,
        qkv_bias=True,
        with_fpn=False,
        frozen_stages=4,
        drop_path_rate=0,
        out_indices=(0,1,2,3),
        init_cfg=dict(type='Pretrained', checkpoint='path_or_url.pth'),
    ),
    decode_head = dict(
        type='FCNHead',
        in_channels=[1280, 1280, 1280, 1280],
        in_index=[0, 1, 2, 3],
        input_transform='resize_concat',
        channels=5120,
        num_convs=0,
        concat_input=False, 
        dropout_ratio=0, 
        num_classes=21, 
        norm_cfg=norm_cfg,  
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)
        ),
    test_cfg = dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
)

max_iters=10000

# Optimization configurations

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0)
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=200
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        by_epoch=False,
        begin=200,
        end=max_iters
    )
]

# Training loop configurations
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=max_iters+1,
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=100,
        log_metric_by_epoch=False
    ),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1000,
        max_keep_ckpts=1,
        save_best='mIoU',
        rule='greater'
    ),
    timer=dict(type='IterTimerHook')
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator