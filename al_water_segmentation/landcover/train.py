import argparse
import logging
import os
import os.path as osp
import torch
import numpy as np

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.datasets import build_dataset
from torch.utils.data import DataLoader

from mmseg.registry import RUNNERS
from mmseg.apis import inference_segmentor
from mmseg.models import build_segmentor
#from mmseg.datasets import build_dataloader

# Мы должны импортировать ваш датасет
from dataset import LandcoverAI  # Убедитесь, что этот датасет зарегистрирован

# Функция для вычисления энтропии предсказаний
def calculate_entropy(probabilities):
    """Вычисляет энтропию для каждого пикселя."""
    return -np.sum(probabilities * np.log(probabilities + 1e-6), axis=-1)

def select_uncertain_samples(dataloader, model, num_samples=5):
    """Выбирает наиболее неопределенные изображения для разметки."""
    uncertain_samples = []
    for idx, data in enumerate(dataloader):
        img = data['img'][0].unsqueeze(0).cuda()  # Извлекаем изображение
        # Получаем вероятностное распределение от модели
        result = inference_segmentor(model, img)
        probs = torch.softmax(torch.tensor(result), dim=1).cpu().numpy()  # Преобразуем в вероятности
        
        # Рассчитываем энтропию для каждого пикселя
        entropy_map = calculate_entropy(probs)
        
        # Суммируем энтропию по всем пикселям изображения
        entropy_score = np.sum(entropy_map)
        
        uncertain_samples.append((idx, entropy_score))
        
        if len(uncertain_samples) >= num_samples:
            break
            
    # Сортируем по энтропии (чем больше, тем хуже модель уверена)
    uncertain_samples.sort(key=lambda x: x[1], reverse=True)
    return uncertain_samples

def update_and_train_model(new_data, model, train_dataloader, cfg):
    """Добавить новые данные для обучения и переобучить модель"""
    # Здесь предполагается, что вы добавляете новые данные в тренировочный набор
    # Для этого нужно обновить пути к данным в train_dataloader
    train_dataloader['dataset']['data_prefix']['img_path'] = new_data['img_path']
    train_dataloader['dataset']['data_prefix']['seg_map_path'] = new_data['seg_map_path']

    # Переобучение модели
    runner = Runner.from_cfg(cfg)
    runner.train()

def active_learning(cfg):
    """Основной цикл активного обучения"""
    # Создаем модель
    model = build_segmentor(cfg.model)

    # Создаем тренировочный датасет
    train_dataset = build_dataset(cfg.data.train)

    # Создаем валидационный датасет
    val_dataset = build_dataset(cfg.data.val)

    # Создаем даталоадеры
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.data.samples_per_gpu,  # или используйте batch_size из конфигурации
        shuffle=True,
        num_workers=cfg.data.workers_per_gpu
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.data.samples_per_gpu,  # или используйте batch_size из конфигурации
        shuffle=False,
        num_workers=cfg.data.workers_per_gpu
    )

    for epoch in range(cfg.train_cfg.max_epochs):
        # Обучаем модель на текущем наборе данных
        runner = Runner.from_cfg(cfg)
        runner.train()

        # Выбираем наиболее неопределенные данные
        uncertain_samples = select_uncertain_samples(train_dataloader, model, num_samples=10)

        # Получаем новые данные для разметки (например, вручную или автоматически размеченные)
        new_data = get_new_data(uncertain_samples)  # Реализуйте логику получения новых данных для разметки

        # Обновляем данные и переобучаем модель
        update_and_train_model(new_data, model, train_dataloader, cfg)

        # Периодическая валидация модели
        val_result = model.evaluate(val_dataloader)
        print(f"Validation result at epoch {epoch}: {val_result}")

def parse_args():
    """Парсим аргументы командной строки"""
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
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--active-learning', action='store_true', help="Enable active learning")
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    """Главная функция для тренировки модели"""
    args = parse_args()

    # Загружаем конфиг
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir задается в конфиге или через CLI
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # Включаем автоматическое смешанное вычисление, если нужно
    if args.amp:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log('AMP training is already enabled in your config.', logger='current', level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # Если включено активное обучение
    if args.active_learning:
        active_learning(cfg)  # Включаем активное обучение

    else:
        # Стандартный тренажер
        runner = Runner.from_cfg(cfg)
        runner.train()


if __name__ == '__main__':
    main()
