# miners.py

import os
import os.path as osp
import random
import torch
import numpy as np
import json

from mmengine.utils import ProgressBar
from mmengine.evaluator import BaseMetric
from mmengine.registry import Registry

# Ваш собственный реестр MINERS, например:
from .registry import MINERS


def al_scores_single_gpu(model, data_loader, logger, save_dir, active_cycle=-1, **kwargs):
    """
    Пример функции, считающей «score» (неуверенность) для 
    каждого изображения из data_loader на одном GPU.

    :param model: Модель (runner.model), которую будем вызывать.
    :param data_loader: Даталоадер, пробегающийся по «unlabeled» данным.
    :param logger: Логгер для вывода прогресса.
    :param save_dir: Папка, куда можно писать результаты (если нужно).
    :param active_cycle: Номер итерации AL (по желанию).
    :param kwargs: Доп. аргументы (не обязательны).

    :return: Словарь results = { 'img_id': [...], 'score': [...], 'meta': [...] }
    """
    model.eval()
    results = dict(img_id=[], score=[], meta=[])

    logger.info(f"Acquiring Active Learning Scores (cycle={active_cycle})")
    progress_bar = ProgressBar(len(data_loader))

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # Прогоним data через data_preprocessor модели
            data = model.data_preprocessor(data, False)
            # Допустим, forward(mode='active') возвращает словари:
            #   {'img_id': [...], 'score': [...], 'meta': [...]} 
            result = model.forward(**data, mode='active')

        # Копим результаты
        for key in ['img_id', 'score', 'meta']:
            results[key].extend(result[key])

        progress_bar.update()

    return results


@MINERS.register_module()
class JsonWriter:
    """
    Класс, создающий/обновляющий `labeled.json` и `unlabeled.json` 
    на каждом шаге активного обучения.

    Предполагаем формат «COCO-подобный»:
      {
        "images": [...],
        "annotations": [...],
        "info": {...},
        "categories": [...],
        "license": [...optional...]
      }
    где у каждого изображения есть "id", а у каждой аннотации — "image_id".
    """

    def __init__(self,
                 data_file,
                 initial_labeled_size=1000,
                 labeled_size=1000):
        """
        :param data_file: Путь к единому JSON (COCO-подобному), из которого берём 
                          все изображения/аннотации.
        :param initial_labeled_size: Сколько изображений класть в labeled.json при первом цикле.
        :param labeled_size: Сколько «сложных» изображений добавлять в labeled.json 
                             на следующих итерациях.
        """
        self.labeled_size = labeled_size
        self.initial_labeled_size = initial_labeled_size
        self.logger = None

        # Путь к исходному JSON
        self.data_file = data_file

        # Данные из data_file, прочитаем при первом обращении
        self.dataset_json = None

        # Для хранения идентификаторов
        self.total_images = set()
        self.labeled_set = set()
        self.unlabeled_set = set()

        # Номер итерации AL (устанавливается извне)
        self.al_cycle = 0

    def set_logger(self, logger):
        """Получаем логгер от Runner, если хотим выводить информацию."""
        self.logger = logger

    def create_initial_data_partition(self, save_dir):
        """
        Формируем стартовое разделение: "labeled.json" и "unlabeled.json"
        с `initial_labeled_size` случайными «размеченными» изображениями.
        Остальные — «неразмеченные».
        """
        with open(self.data_file, 'r') as f:
            self.dataset_json = json.load(f)

        images, annotations, info, categories, license_ = self._json_separate(self.dataset_json)
        self.total_images = set(img['id'] for img in images)

        # Случайно выбираем initial_labeled_size
        # (убедитесь, что total_images >= initial_labeled_size, иначе будет ошибка)
        self.labeled_set.update(random.sample(self.total_images, self.initial_labeled_size))
        self.unlabeled_set.update(self.total_images - self.labeled_set)

        # Создаём два JSON
        labeled_json, unlabeled_json = self._create_labeled_unlabeled(
            images, annotations, info, categories, license_
        )

        labeled_json_file = osp.join(save_dir, "labeled.json")
        unlabeled_json_file = osp.join(save_dir, "unlabeled.json")

        with open(labeled_json_file, 'w') as f:
            json.dump(labeled_json, f, indent=2)
        with open(unlabeled_json_file, 'w') as f:
            json.dump(unlabeled_json, f, indent=2)

        if self.logger is not None:
            self.logger.info(
                f"[cycle={self.al_cycle}] Saved labeled/unlabeled json at {save_dir}"
            )
            self.logger.info(
                f"Total images: {len(self.total_images)} | "
                f"Labeled: {len(self.labeled_set)} | Unlabeled: {len(self.unlabeled_set)}"
            )

    def create_data_partitions(self, scores_file, save_dir):
        """
        При следующих итерациях AL: читаем scores.json, выбираем top-K самых «сложных»
        изображений (по score), добавляем их в labeled_set и обновляем labeled.json/unlabeled.json.
        """
        if self.logger is not None:
            self.logger.info(f"Load AL scores from {scores_file}")

        with open(scores_file, 'r') as f:
            scores_json = json.load(f)

        scores = np.array(scores_json['score'])
        img_ids = np.array(scores_json['img_id'])

        # Выбираем top-N (self.labeled_size) самых «сложных»
        # idx_top — индексы самых больших score
        idx_top = np.argpartition(-scores, self.labeled_size - 1)[:self.labeled_size]
        top_img_ids = set(img_ids[idx_top].tolist())

        # Переносим их из unlabeled -> labeled
        self.labeled_set.update(top_img_ids)
        self.unlabeled_set -= top_img_ids

        # Формируем новые labeled/unlabeled
        images, annotations, info, categories, license_ = self._json_separate(self.dataset_json)
        labeled_json, unlabeled_json = self._create_labeled_unlabeled(
            images, annotations, info, categories, license_
        )

        labeled_json_file = osp.join(save_dir, "labeled.json")
        unlabeled_json_file = osp.join(save_dir, "unlabeled.json")

        with open(labeled_json_file, 'w') as f:
            json.dump(labeled_json, f, indent=2)
        with open(unlabeled_json_file, 'w') as f:
            json.dump(unlabeled_json, f, indent=2)

        if self.logger is not None:
            self.logger.info(
                f"[cycle={self.al_cycle}] Updated labeled/unlabeled at {save_dir}"
            )
            self.logger.info(
                f"Total images: {len(self.total_images)} | "
                f"Labeled: {len(self.labeled_set)} | Unlabeled: {len(self.unlabeled_set)}"
            )

    def _create_labeled_unlabeled(self, images, annotations, info, categories, license_):
        """
        Формируем два словаря:
          labeled.json, unlabeled.json
        на основе labeled_set / unlabeled_set
        """
        labeled_json = dict(
            images=[], 
            annotations=[],
            info=info,
            categories=categories,
            license=license_
        )
        unlabeled_json = dict(
            images=[],
            annotations=[],
            info=info,
            categories=categories,
            license=license_
        )

        # Разделяем images
        for img in images:
            if img['id'] in self.labeled_set:
                labeled_json['images'].append(img)
            else:
                unlabeled_json['images'].append(img)

        # Разделяем annotations по image_id
        for ann in annotations:
            if ann['image_id'] in self.labeled_set:
                labeled_json['annotations'].append(ann)
            else:
                unlabeled_json['annotations'].append(ann)

        return labeled_json, unlabeled_json

    def _json_separate(self, dataset_json):
        """
        Извлекаем ключи:
          'images', 'annotations', 'info', 'categories', 'license'
        из общего COCO-подобного JSON.
        """
        return (
            dataset_json.get('images', []),
            dataset_json.get('annotations', []),
            dataset_json.get('info', {}),
            dataset_json.get('categories', []),
            dataset_json.get('license', []),
        )
