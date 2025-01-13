_base_ = [
    '../../configs/_base_/models/fpn_poolformer_s12.py',
    '../../configs/_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmseg.datasets.transforms'],  # чтобы RandomCrop и др. зарегистрировались
    allow_failed_imports=False
)

dataset_type = 'LandcoverAI'  # <-- используем LandcoverAL (один класс)

crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(2048, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='ResizeToMultiple', size_divisor=32),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

model = dict(
    data_preprocessor=dict(size=crop_size),
    backbone=dict(
        arch='m48',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/...poolformer.pth',
            prefix='backbone.'
        )
    ),
    neck=dict(in_channels=[96, 192, 384, 768])
)

# Пример оптимизатора, шедулера
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=3e-4, weight_decay=0.001),
)

# Dataloaders
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type='LandcoverAI',
        data_root='/misc/home1/m_imm_freedata/Segmentation/Projects/mmseg_water/landcover.ai_512',  
        ann_file='dataset_coco.json',      # <-- Пусть по умолчанию
        pipeline=train_pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=True)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='/misc/home1/m_imm_freedata/Segmentation/Projects/mmseg_water/landcover.ai_512',
        ann_file='val.json',   # или val/images, val/gt и т.д.
        data_prefix=dict(img_path='val/images', seg_map_path='val/gt'),
        pipeline=test_pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1, save_best='mIoU'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=True)
)

experiment_name = 'AL_experiment'

experiment_name = 'LandcoverAI'
json_writer = dict(
    type='JsonWriter',
    data_file='dataset_coco.json',
    initial_labeled_size=200,
    labeled_size=100
)

active_learning_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root='/misc/home1/m_imm_freedata/Segmentation/Projects/mmseg_water/landcover.ai_512',
        ann_file='unlabeled.json',  # будет подменяться при AL
        data_prefix=dict(img_path='train/images', seg_map_path='train/gt'),
        pipeline=test_pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)
