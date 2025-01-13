_base_ = [
    '../../configs/_base_/models/fpn_poolformer_s12.py',
    '../../configs/_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmseg.datasets.transforms'],  # чтобы RandomCrop и др. зарегистрировались
    allow_failed_imports=False
)


dataset_type = 'LandcoverAI'  # <-- используем LandcoverAL (один класс)

#data_root = '/path/to/data_root'
crop_size = (512, 512)

norm_cfg = dict(type='BN', requires_grad=True)

# Название датасета из файла urfu_project/dataset.py
dataset_type = 'LandcoverAI'

# Путь к папке с преобразованным набором данных
data_root = '/misc/home1/m_imm_freedata/Segmentation/Projects/mmseg_water/landcover.ai_512'

# Количество классов для сегментации
num_classes = 2

# Размер изображения, который принимает на вход сеть
crop_size = (512, 512)

# Количичество эпох для обучения
max_epochs = 100

# Функция потерь
loss = dict(type='FocalLoss', class_weight=[0.9, 1.1])

# Размер батча
batch_size = 16
gradient_accumulation_steps = 8
actual_batch_size = batch_size * gradient_accumulation_steps

# num_workers
num_workers = 8

# Оптимизатор
# optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0005)
optimizer = dict(type='AdamW', lr=3e-4, weight_decay=0.001)

# Параметры логирования 
experiment_name = f'Poolformer_{dataset_type}_{crop_size[0]}_{loss["type"]}_{optimizer["type"]}_bsize_{actual_batch_size}'
logs_dir = 'logs'
work_dir = f'{logs_dir}/{experiment_name}'  # директория для сохранения логов
log_interval = 10  # интервал в итерациях для печати логов

# Директория, где хрянятся файлы с списком изображений train и val
splits = 'splits'

# Имя эксперимента
# ----------------------------------------------------------------
# Параметры модели
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

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

checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-m48_3rdparty_32xb128_in1k_20220414-9378f3eb.pth'  # noqa

# model settings
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


# ----------------------------------------------------------------
# Параметры default_runtime
log_processor = dict(by_epoch=True)
# ----------------------------------------------------------------
# Параметры scheduler
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=None,
    accumulative_counts=gradient_accumulation_steps,
)
# learning policy
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
# В default_hooks draw=False для того, чтобы не выводить изображения из val с результатами модели после эпохи в логи
# Если поставить True, то увеличьте интервал до более высокого, чтоб не сохранял часто много изображений
default_hooks = dict(logger=dict(type='LoggerHook', log_metric_by_epoch=True, interval=log_interval),
                     checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1, max_keep_ckpts=1, save_best='mIoU'),
                     visualization=dict(type='SegVisualizationHook', draw=False, interval=500))



# Dataloaders
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type='LandcoverAI',
        data_root='/misc/home1/m_imm_freedata/Segmentation/Projects/mmseg_water/landcover.ai_512',  
        ann_file='train_coco.json',      # <-- Пусть по умолчанию
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
        ann_file='val_coco.json',   # или val/images, val/gt и т.д.
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
