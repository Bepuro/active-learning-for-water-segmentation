_base_ = [
    '../../configs/_base_/models/fpn_poolformer_s12.py',
    '../../configs/_base_/default_runtime.py',
]


custom_imports = dict(
    imports=['mmseg.datasets.transforms'],
    allow_failed_imports=False
)

norm_cfg = dict(type='BN', requires_grad=True)
dataset_type = 'LandcoverAI'

# Путь к папке с набором данных
data_root = '/misc/home1/m_imm_freedata/Segmentation/Projects/mmseg_water/landcover.ai_512'

num_classes = 2
crop_size = (512, 512)
max_epochs = 100

loss = dict(type='FocalLoss', class_weight=[0.9, 1.1])

batch_size = 16
gradient_accumulation_steps = 8
actual_batch_size = batch_size * gradient_accumulation_steps

num_workers = 8

optimizer = dict(type='AdamW', lr=3e-4, weight_decay=0.001)

experiment_name = f'Poolformer_{dataset_type}_{crop_size[0]}_{loss["type"]}_{optimizer["type"]}_bsize_{actual_batch_size}'
logs_dir = 'logs'
work_dir = f'{logs_dir}/{experiment_name}'
log_interval = 10
splits = 'splits'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_preprocessor = dict(size=crop_size)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    #dict(type='RandomCrop',crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    #dict(type='PhotoMetricDistortion'),
    #dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='ResizeToMultiple', size_divisor=32),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-m48_3rdparty_32xb128_in1k_20220414-9378f3eb.pth'

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        arch='m48',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint_file,
            prefix='backbone.'
        )
    ),
    neck=dict(in_channels=[96, 192, 384, 768])
)

log_processor = dict(by_epoch=True)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=None,
    accumulative_counts=gradient_accumulation_steps,
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=3e-2,
        begin=0,
        end=45,
        by_epoch=True,
    ),
    dict(
        type='PolyLRRatio',
        eta_min_ratio=3e-2,
        power=0.9,
        begin=45,
        end=90,
        by_epoch=True,
    ),
    dict(
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=90,
        end=100,
    )
]

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
train_cfg = dict(max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=1)

default_hooks = dict(
    logger=dict(type='LoggerHook', log_metric_by_epoch=True, interval=log_interval),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1, max_keep_ckpts=1, save_best='mIoU'),
    visualization=dict(type='SegVisualizationHook', draw=False, interval=500)
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/images',
            seg_map_path='train/gt'
        ),
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/images',
            seg_map_path='val/gt'
        ),
        pipeline=test_pipeline,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

vis_backends = [
    dict(type='LocalVisBackend', scalar_save_file='../../scalars.json', save_dir=work_dir),
    dict(type='TensorboardVisBackend', save_dir=work_dir)
]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
