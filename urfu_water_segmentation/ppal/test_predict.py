#!/usr/bin/env python
import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import logging
import time
from pathlib import Path
import torch

from mmseg.apis import MMSegInferencer

# Константы для разметки
WATER_PIXEL_VALUE = 64
BACKGROUND_CLASS_IDX = 0
WATER_CLASS_IDX = 1

# Палитра для визуализации (например, 0 – фон, 64 – вода)
PALETTE = [
    [0, 0, 0],
    [64, 64, 64],
]

def find_config_path(experiment_path: Path) -> Path:
    """Найти путь к конфигурационному файлу (.py) внутри папки эксперимента."""
    return next(experiment_path.glob("*.py"))

def calculate_iou_and_accuracy(predicted, mask):
    """Вычисление IoU и accuracy для бинарной маски."""
    eps = 1e-8
    # Определяем количество пикселей, где предсказание совпадает с маской
    intersect = predicted[predicted == mask]
    area_intersect, _ = np.histogram(intersect, bins=2, range=(0, 1))
    area_pred, _ = np.histogram(predicted, bins=2, range=(0, 1))
    area_gt, _ = np.histogram(mask, bins=2, range=(0, 1))
    area_union = area_pred + area_gt - area_intersect
    iou = (area_intersect + eps) / (area_union + eps)
    acc = (area_intersect + eps) / (area_gt + eps)
    return iou, acc

def return_difficulties(output_path, experiment_path, images):
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    
    try:
        # Логирование начала работы
        logger.info("Starting inference process...")
        logger.info(f"Processing {len(images)} images")
        
        config_path = str(find_config_path(experiment_path))
        logger.info(f"Config path: {config_path}")
        
        weights_file = experiment_path / "last_checkpoint"
        weights_path = weights_file.read_text().strip()
        logger.info(f"Weights path: {weights_path}")
        
        # Логирование инициализации модели
        logger.info("Initializing inferencer...")
        init_start = time.time()
        
        inferencer = MMSegInferencer(
            model=config_path,
            weights=weights_path,
            device="cuda" if torch.cuda.is_available() else "cpu",  # Используем GPU если доступен
        )
        
        logger.info(f"Inferencer initialized in {time.time() - init_start:.2f} seconds")
        
        # Логирование процесса инференса
        logger.info("Starting inference...")
        infer_start = time.time()
        
        # Разбиваем на батчи с прогресс-баром
        results = []
        batch_size = 4  # Можно настроить оптимальный размер батча
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(images)-1)//batch_size + 1}")
            
            batch_start = time.time()
            batch_results = inferencer(batch, batch_size=batch_size)
            #print(batch_results["predictions"])
            predictions = batch_results.get("predictions", [])
            
            # Добавляем каждый словарь из predictions в results
            results.extend(predictions)
            
            logger.info(f"Batch processed in {time.time() - batch_start:.2f} seconds")
        
        total_time = time.time() - start_time
        logger.info(f"Inference completed in {total_time:.2f} seconds")
        logger.info(f"Processed {len(images)} images ({len(images)/total_time:.2f} img/s)")
        #print(type(results[0]))
        
        return results
    
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise

def write_plots_and_visualize(path_to_res, **images):
    """
    Функция для визуализации: располагает переданные изображения в ряд и сохраняет результат.
    """
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        if image is not None:
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)
    plt.savefig(path_to_res, bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser(
        description="Инференс для одного изображения с вычислением метрик."
    )
    parser.add_argument("--image", required=True, help="Путь к входному изображению")
    parser.add_argument("--gt", required=False, help="Путь к ground truth маске (опционально)")
    parser.add_argument("-o", "--output-path", required=True, help="Папка для сохранения результатов")
    parser.add_argument("-e", "--experiment-path", required=True,
                        help="Путь к папке эксперимента (с конфигурацией и файлом last_checkpoint)")
    parser.add_argument("--visualize", action="store_true", help="Визуализировать результат")
    parser.add_argument("--device", default=None, help="Устройство для инференса (например, 'cpu' или 'cuda')")
    
    args = parser.parse_args()
    
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    experiment_path = Path(args.experiment_path)
    config_path = str(find_config_path(experiment_path))
    # Читаем путь к весам из файла last_checkpoint (убираем лишние пробелы и символы переноса)
    weights_path = (experiment_path / "last_checkpoint").read_text().strip()
    
    # Инициализация инференсера
    inferencer = MMSegInferencer(
        model=config_path,
        weights=weights_path,
        device=args.device,
    )
    
    # Выполняем инференс для одного изображения
    result = inferencer([args.image], batch_size=1)
    print(result)
    # prediction = result["predictions"][0]

    prediction = result["predictions"]["sem_seg"]
    print(result)

    metrics = {}

    if args.gt:
        gt = cv2.imread(args.gt, cv2.IMREAD_GRAYSCALE)
        # Приводим GT к бинарной маске: 0 для фона, 1 для воды
        gt[gt != WATER_PIXEL_VALUE] = BACKGROUND_CLASS_IDX
        gt[gt == WATER_PIXEL_VALUE] = WATER_CLASS_IDX
        
        iou, acc = calculate_iou_and_accuracy(prediction, gt)
        # Приводим результаты к читаемому виду
        metrics["IoU"] = iou.tolist() if hasattr(iou, "tolist") else iou
        metrics["Accuracy"] = acc.tolist() if hasattr(acc, "tolist") else acc
    
    # Сохраняем предсказанную маску в виде изображения
    pred_img = Image.fromarray(prediction.astype(np.uint8)).convert("P")
    pred_img.putpalette(np.array(PALETTE, dtype=np.uint8))
    pred_save_path = output_path / "prediction.png"
    pred_img.save(pred_save_path)
    
    # Если требуется визуализация, формируем компоновку изображений
    if args.visualize:
        original_image = Image.open(args.image)
        vis_images = {
            "original": original_image,
            "prediction": np.array(pred_img)
        }
        if args.gt:
            vis_images["ground_truth"] = gt
        vis_save_path = output_path / "visualization.png"
        write_plots_and_visualize(str(vis_save_path), **vis_images)
    
    print("Предсказание сохранено:", pred_save_path)

    if metrics:
        print("Вычисленные метрики:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
    
if __name__ == "__main__":
    main()