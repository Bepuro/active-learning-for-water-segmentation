import os
import os.path as osp
import random
import torch
import numpy as np
import json
import argparse
from torch.utils.data import DataLoader
from dataset import LandcoverAI
from mmseg.models import build_segmentor
from mmengine.config import Config

class JsonWriter:
    def __init__(self, data_file, initial_labeled_size=1000, labeled_size=1000):
        self.labeled_size = labeled_size
        self.initial_labeled_size = initial_labeled_size
        self.logger = None
        self.data_file = data_file
        self.dataset_json = None
        self.total_images = set()
        self.labeled_set = set()
        self.unlabeled_set = set()
        self.al_cycle = 0

    def set_logger(self, logger):
        self.logger = logger

    def create_initial_data_partition(self, save_dir):
        with open(self.data_file, 'r') as f:
            self.dataset_json = json.load(f)

        images, annotations, info, categories, license_ = self._json_separate(self.dataset_json)
        self.total_images = set(img['id'] for img in images)
        self.labeled_set.update(random.sample(list(self.total_images), self.initial_labeled_size))
        self.unlabeled_set.update(self.total_images - self.labeled_set)

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
            self.logger.info(f"[cycle={self.al_cycle}] Saved labeled/unlabeled json at {save_dir}")

    def _create_labeled_unlabeled(self, images, annotations, info, categories, license_):
        labeled_json = dict(images=[], annotations=[], info=info, categories=categories, license=license_)
        unlabeled_json = dict(images=[], annotations=[], info=info, categories=categories, license=license_)

        for img in images:
            if img['id'] in self.labeled_set:
                labeled_json['images'].append(img)
            else:
                unlabeled_json['images'].append(img)

        for ann in annotations:
            if ann['image_id'] in self.labeled_set:
                labeled_json['annotations'].append(ann)
            else:
                unlabeled_json['annotations'].append(ann)

        return labeled_json, unlabeled_json

    def _json_separate(self, dataset_json):
        return (
            dataset_json.get('images', []),
            dataset_json.get('annotations', []),
            dataset_json.get('info', {}),
            dataset_json.get('categories', []),
            dataset_json.get('license', []),
        )

def build_dataloader(dataset, batch_size, num_workers, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def calculate_dcus_uncertainty(logits, iou_scores, class_difficulties, alpha=0.5):
    probs = torch.softmax(logits, dim=1)
    entropies = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    difficulty_weights = 1 + alpha * class_difficulties
    uncertainty = difficulty_weights * (entropies * (1 - iou_scores))
    return uncertainty

def al_scores_single_gpu(model, data_loader, logger, save_dir, active_cycle=-1, class_difficulties=None, **kwargs):
    model.eval()
    results = dict(img_id=[], score=[], meta=[])

    logger.info(f"Acquiring Active Learning Scores (cycle={active_cycle})")

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            processed_data = model.data_preprocessor(data, False)
            outputs = model.forward(**processed_data, mode='active')
            logits = outputs['logits']
            iou_scores = outputs.get('iou_scores', torch.ones_like(logits[:, 0]))
            uncertainty = calculate_dcus_uncertainty(logits, iou_scores, class_difficulties)

            for idx, img_meta in enumerate(data['img_meta']):
                results['img_id'].append(img_meta['img_id'])
                results['score'].append(uncertainty[idx].item())
                results['meta'].append(img_meta)

    if save_dir:
        score_file = osp.join(save_dir, f'scores_cycle_{active_cycle}.json')
        with open(score_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved scores to {score_file}")

    return results

if __name__ == "__main__":
    config_path = "./config_landcover.py"
    data_file = "./train_coco.json"
    save_dir = ""
    initial_size = 1000
    labeled_size = 100
    cycles = 3

    class MockLogger:
        def info(self, msg):
            print(msg)

    logger = MockLogger()

    cfg = Config.fromfile(config_path)

    dataset = LandcoverAI(
        ann_file=data_file,
        img_suffix='.tif',
        seg_map_suffix='.png',
        data_root=''
    )

    dataloader = build_dataloader(dataset, batch_size=16, num_workers=4, shuffle=True)

    class_difficulties = torch.ones(3)

    for cycle in range(cycles):
        logger.info(f"Starting Active Learning Cycle {cycle}")
        save_dir_cycle = osp.join(save_dir, f"cycle_{cycle}")
        os.makedirs(save_dir_cycle, exist_ok=True)

        json_writer = JsonWriter(data_file=data_file, initial_labeled_size=initial_size, labeled_size=labeled_size)
        json_writer.set_logger(logger)

        if cycle == 0:
            json_writer.create_initial_data_partition(save_dir_cycle)

        model = build_segmentor(cfg.model)

        al_scores_single_gpu(model, dataloader, logger, save_dir_cycle, active_cycle=cycle, class_difficulties=class_difficulties)

        logger.info(f"Cycle {cycle} completed.")
