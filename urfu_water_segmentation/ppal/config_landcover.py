# для выбора модели, расписания необходимо наследовать один из файлов из репозитория mmsegmentation
# базовые файлы для наследования можно посмотреть по пути mmsegmentation/configs/_base_/
_base_ = [
    '../../configs/_base_/models/fpn_poolformer_s12.py',
    '../../configs/_base_/default_runtime.py',
    # '../../configs/_base_/schedules/schedule_40k.py'
]

# ----------------------------------------------------------------
# Изменение гиперпараметров

# Поскольку мы используем только один графический процессор, вместо SyncBN используется BN
norm_cfg = dict(type='BN', requires_grad=True)

# Название датасета из файла urfu_project/dataset.py
dataset_type = 'ALDataset'

# Путь к папке с преобразованным набором данных
data_root = '/misc/home1/m_imm_freedata/Segmentation/Projects/GLH_water/glh_cut_512_filtered'

# Количество классов для сегментации
num_classes = 2

# Размер изображения, который принимает на вход сеть
crop_size = (512, 512)

# Количичество эпох для обучения
max_epochs = 15

# Функция потерь
loss = dict(type='FocalLoss', class_weight=[0.9, 1.1])

# Размер батча
batch_size = 16
gradient_accumulation_steps = 8
actual_batch_size = batch_size * gradient_accumulation_steps

# num_workers
num_workers = 12

# Оптимизатор
# optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0005)
optimizer = dict(type='AdamW', lr=3e-4, weight_decay=0.001)

# Параметры логирования 
experiment_name = f'GLH_AL_max__Poolformer_{dataset_type}_{crop_size[0]}_{loss["type"]}_{optimizer["type"]}_bsize_{actual_batch_size}'
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
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='ResizeToMultiple', size_divisor=32),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
checkpoint_file = '/misc/home6/s0204/urfu_seg/urfu_water_segmentation/ppal/logs/GLH_AL_max__Poolformer_ALDataset_512_FocalLoss_AdamW_bsize_128/epoch_15.pth'
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
# ----------------------------------------------------------------

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
            seg_map_path='train/gt'),
        pipeline=train_pipeline,
        ann_file='', 
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
            seg_map_path='val/gt'),
        pipeline=test_pipeline,
        # ann_file=f'{splits}/val.txt'
        )
    )

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
# tta_pipeline = [  # Test Time Augmentation (TTA)
#     dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
#     dict(
#         type='TestTimeAug',
#         transforms=[
#             [
#                 dict(type='Resize', scale_factor=r, keep_ratio=True)
#                 for r in img_ratios
#             ],
#             [
#                 dict(type='RandomFlip', prob=0., direction='horizontal'),
#                 dict(type='RandomFlip', prob=1., direction='horizontal')
#             ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
#         ])
# ]
# ----------------------------------------------------------------
# Tensorboard visualization
vis_backends = [dict(type='LocalVisBackend', scalar_save_file='../../scalars.json', save_dir=work_dir),
                dict(type='TensorboardVisBackend', save_dir=work_dir)]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
# ----------------------------------------------------------------
