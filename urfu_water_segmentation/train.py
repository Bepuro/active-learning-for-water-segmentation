import argparse
import os
import os.path as osp
import time

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmengine.registry import RUNNERS
from mmengine.dist import get_dist_info
from mmengine.fileio import dump

# Импортируем функции, написанные в al_mining/utils.py
from al_mining.utils import patch_cfg, patch_runner

# Импортируем наш JsonWriter и функцию подсчёта score (al_scores_single_gpu)
from al_mining.registry import MINERS
from al_mining.miners import JsonWriter, al_scores_single_gpu

# Обязательно импортируем LandcoverAI (в котором ann_file может быть задан из конфига)
from dataset import LandcoverAI


def parse_args():
    parser = argparse.ArgumentParser(description='Train LandcoverAI with Active Learning, storing labeled/unlabeled JSON.')
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('--work-dir', help='Directory to save logs and models')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume training from last checkpoint')
    parser.add_argument('--amp', action='store_true', default=False, help='Enable Automatic Mixed Precision')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='Override config settings')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--al_cycles', type=int, default=3, help='Number of Active Learning cycles')

    args = parser.parse_args()

    # Устанавливаем LOCAL_RANK
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def build_cfg(args):
    """Загружаем и настраиваем конфиг (AMP, work_dir, ...)."""
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.amp:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log('AMP is already enabled in config.', logger='current')
        else:
            assert optim_wrapper == 'OptimWrapper', '--amp only works if the optimizer is `OptimWrapper`.'
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    rank, _ = get_dist_info()
    if rank == 0 and os.environ.get('TIMESTAMP', None) is None:
        os.environ['TIMESTAMP'] = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    if cfg.get('base_work_dir', None) is None:
        cfg.base_work_dir = f"{cfg.work_dir}_{os.environ['TIMESTAMP']}"

    # Сохраняем число AL-циклов
    cfg.al_cycles = args.al_cycles
    # По желанию:
    cfg.resume = args.resume

    return cfg


def main():
    args = parse_args()

    # Цикл по количеству шагов активного обучения
    for al_cycle in range(args.al_cycles):
        cfg = build_cfg(args)
        cfg.al_cycle = al_cycle

        # 1. Патчим конфиг (создаём step_{i}, подменяем ann_file => labeled.json)
        patch_cfg(cfg)
        os.makedirs(cfg.work_dir, exist_ok=True)

        # 2. Сборка Runner
        if 'runner_type' not in cfg:
            runner = Runner.from_cfg(cfg)
        else:
            runner = RUNNERS.build(cfg)

        # 3. Патчим Runner (меняем логи на step_{i} и т.д.)
        patch_runner(runner, cfg)

        # 4. Создаём или обновляем labeled.json/unlabeled.json
        #    через JsonWriter (строим из cfg.json_writer)
        if hasattr(cfg, 'json_writer'):
            json_writer = MINERS.build(cfg.json_writer)
            json_writer.al_cycle = al_cycle
            json_writer.set_logger(runner.logger)

            if al_cycle == 0:
                # Первый цикл: создаём начальную выборку (initial_labeled_size)
                json_writer.create_initial_data_partition(cfg.work_dir)
            else:
                # Последующие циклы: читаем scores.json, переносим top-K => labeled
                scores_file = osp.join(cfg.base_work_dir, f'step_{al_cycle-1}', 'scores.json')
                json_writer.create_data_partitions(scores_file=scores_file, save_dir=cfg.work_dir)
        else:
            runner.logger.warning('No `json_writer` found in config, skipping labeled/unlabeled partitioning.')

        # 5. Запускаем обучение (теперь dataloader читает ann_file=step_{i}/labeled.json)
        runner.train()

        # 6. Считаем score для неразмеченных (если нужно)
        if hasattr(cfg, 'active_learning_dataloader'):
            # Собираем unlabeled_dataloader
            unlabeled_dataloader = Runner.build_dataloader(cfg.active_learning_dataloader)
            results = al_scores_single_gpu(runner.model, unlabeled_dataloader,
                                           runner.logger, cfg.work_dir, active_cycle=al_cycle)
            # Сохраняем scores.json (чтобы на следующем шаге добавить top-K в labeled.json)
            score_file = osp.join(cfg.work_dir, 'scores.json')
            runner.logger.info(f'Saving scores.json to {score_file}')
            dump(results, score_file)

if __name__ == '__main__':
    main()
