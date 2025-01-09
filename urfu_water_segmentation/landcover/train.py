# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import random  # нужно для случайного перемешивания

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS, DATASETS




# Импортируем наш кастомный датасет (LandcoverAI)
from dataset import LandcoverAI


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # Когда PyTorch >= 2.0.0 используется torch.distributed.launch,
    # оно будет передавать '--local-rank', а не '--local_rank'
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    # Флаг для запуска режима активного обучения
    parser.add_argument(
        '--active-learning',
        action='store_true',
        default=False,
        help='Enable Active Learning loop')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def active_learning_loop(cfg):
    """Упрощённый пример цикла активного обучения с автоделением датасета."""

    # Собираем единый датасет из описания (все данные размечены):
    dataset_full = DATASETS.build(cfg.train_dataloader.dataset)

    # Определяем, какую долю оставим размеченной (например, 10%)
    labeled_fraction = 0.1
    data_list = dataset_full.data_list[:]  # скопируем список
    random.shuffle(data_list)
    num_labeled = int(len(data_list) * labeled_fraction)

    # Разбиваем на "размеченный" (labeled) и "неразмеченный" (unlabeled) по data_list
    labeled_data_list = data_list[:num_labeled]
    unlabeled_data_list = data_list[num_labeled:]

    # Создаём два объекта датасета с разными data_list
    labeled_dataset = DATASETS.build(cfg.train_dataloader.dataset)
    labeled_dataset.data_list = labeled_data_list

    unlabeled_dataset = DATASETS.build(cfg.train_dataloader.dataset)
    unlabeled_dataset.data_list = unlabeled_data_list

    # Строим DataLoader для размеченного датасета (на котором реально будем обучаться)
    labeled_loader_cfg = cfg.train_dataloader.copy()
    labeled_loader_cfg['dataset'] = labeled_dataset
    labeled_loader_cfg['sampler'] = dict(type='DefaultSampler', shuffle=True)
    labeled_dataloader = Runner.build_dataloader(labeled_loader_cfg)

    # Готовим Runner
    runner = Runner.from_cfg(cfg)

    # Цикл активного обучения (3 итерации для примера)
    num_al_iterations = 3
    top_k = 10  # сколько самых "неуверенных" примеров переносить в каждом шаге

    for al_iter in range(1, num_al_iterations + 1):
        print_log(
            f'\n[Active Learning] Start iteration {al_iter}/{num_al_iterations}',
            logger='current', level=logging.INFO)

        # Тренируемся на размеченной части
        state_dict = runner.model.state_dict()

        # Создаем новый runner (уже с новой конфигурацией)
        new_runner = Runner.from_cfg(cfg)

        # Загружаем веса в модель нового Runner
        new_runner.model.load_state_dict(state_dict)

        # Запускаем тренировку заново
        new_runner.train()  # один полный цикл обучения на N эпох (указанных в cfg.train_cfg)

        # Оценка неуверенности на "неразмеченной" части
        uncertainty_scores = compute_uncertainty(unlabeled_dataset, runner)

        # Выбираем top_k самых неуверенных
        sorted_by_unc = sorted(uncertainty_scores, key=lambda x: x[1], reverse=True)
        top_uncertain = sorted_by_unc[:top_k]

        # "Размечаем" их (переносим из unlabeled_dataset в labeled_dataset)
        print_log(
            f'[Active Learning] Adding {top_k} most uncertain samples to the labeled set',
            logger='current', level=logging.INFO)
        move_samples_to_labeled(labeled_dataset, unlabeled_dataset, top_uncertain)

        # Убедимся, что DataLoader знает о новом кол-ве данных
        labeled_loader_cfg['dataset'] = labeled_dataset
        labeled_dataloader = Runner.build_dataloader(labeled_loader_cfg)

        # Если unlabeled_dataset опустел, завершаем AL
        if len(unlabeled_dataset) == 0:
            print_log('[Active Learning] Unlabeled pool is empty.', logger='current', level=logging.INFO)
            break

    print_log('[Active Learning] Finished all AL iterations.', logger='current', level=logging.INFO)


def compute_uncertainty(dataset, runner):
    """Пример функции, считающей «неуверенность» модели для каждого изображения.
    Возвращает список кортежей (idx, score).
    """
    results = []
    for idx in range(len(dataset)):
        # В реальности здесь нужно выполнить forward модели runner.model,
        # посчитать энтропию / дисперсию / etc. Мы же генерируем случайно.
        score = random.random()
        results.append((idx, score))
    return results


def move_samples_to_labeled(labeled_dataset, unlabeled_dataset, top_uncertain):
    """Переносит выбранные индексы из unlabeled_dataset в labeled_dataset."""
    from mmseg.datasets import BaseSegDataset

    if not isinstance(labeled_dataset, BaseSegDataset):
        raise TypeError('labeled_dataset should be an instance of BaseSegDataset.')
    if not isinstance(unlabeled_dataset, BaseSegDataset):
        raise TypeError('unlabeled_dataset should be an instance of BaseSegDataset.')

    indices_to_add = [x[0] for x in top_uncertain]
    new_samples = []
    # Важно вынимать из data_list по убыванию индексов, чтобы не сбить нумерацию
    for idx in sorted(indices_to_add, reverse=True):
        data_info = unlabeled_dataset.data_list.pop(idx)
        new_samples.append(data_info)

    # Считаем, что "новые образцы" как бы разметили.
    labeled_dataset.data_list.extend(new_samples)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log('AMP training is already enabled in your config.',
                      logger='current', level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    cfg.resume = args.resume

    with open(f'{cfg.data_root}/mean_vals.txt', 'r') as f:
        lines = f.readlines()
    mean = [float(x) for x in lines[0].strip().split()]
    std = [float(x) for x in lines[1].strip().split()]
    cfg.model.data_preprocessor.mean = mean
    cfg.model.data_preprocessor.std = std

    if args.active_learning:
        active_learning_loop(cfg)
    else:
        if 'runner_type' not in cfg:
            runner = Runner.from_cfg(cfg)
        else:
            runner = RUNNERS.build(cfg)
        runner.train()


if __name__ == '__main__':
    main()
