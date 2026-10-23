

_base_ = [
    'cascade_mask_rcnn_vit_fpn.py', '../datasets/voc2007.py', 'mmdet::_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['jea.eval.object_detection.register_backbone'],
    allow_failed_imports=False
)

find_unused_parameters=True

model = dict(
    backbone=dict(
        embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        drop_path_rate=0,
        frozen_stages=4,
        ffn_layer="swiglufused",
        init_values=0.1,
        block_chunks=4,
        out_with_norm=True,
    ),
    neck=dict(in_channels=[384, 384, 384, 384]),
    roi_head=dict(
        mask_roi_extractor=None,
        mask_head=None,
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=20,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=20,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=20,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ]),
        
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05)
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=20)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

lr_warmup = dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500)
lr_scheduler = dict(type='MultiStepLR', milestones=[8, 11], gamma=0.1, by_epoch=True)

param_scheduler = [lr_warmup, lr_scheduler]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1,
        save_last=True,
    )
)