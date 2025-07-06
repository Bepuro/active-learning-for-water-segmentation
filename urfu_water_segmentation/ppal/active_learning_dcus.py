import os
import pandas as pd
import subprocess
import shutil
import json
import math
from test_predict import return_difficulties
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 

PALETTE = [
    [0, 0, 0],
    [64, 64, 64],
]


def get_prediction():
    unlabeled_csv = "./data/glh_water/cache/unlabeled.csv"
    output_results = "./data/glh_water/cache/output_results"
    exp_dir = "./logs/GLH_AL_max__Poolformer_ALDataset_512_FocalLoss_AdamW_bsize_128"
    data_mask = "./data/glh_water/cache/mask"
    data_folder = "./data/glh_water/cache/"

    os.makedirs(output_results, exist_ok=True)
    os.makedirs(data_mask, exist_ok=True)

    df_unlabeled = pd.read_csv(unlabeled_csv)

    alpha = 0.8
    beta = 0.4
    gamma = math.e**(1/alpha) - 1

    image_paths = df_unlabeled['img_path'].tolist()
    image_ids = df_unlabeled['index'].tolist()
    outputs = return_difficulties(Path(output_results), Path(exp_dir), image_paths)


    results = []

    for idx, (image_id, img_path, output) in enumerate(zip(image_ids, image_paths, outputs)):
        
        difficulty = output.get("difficulty", None)
        average_confidence = output.get("average_confidence", None)

        if difficulty is not None:
            coefficient = 1 + alpha * beta * math.log(1 + gamma * difficulty)
        else:
            coefficient = None

        new_mask_name = f"{image_id}.png"
        new_mask_path = os.path.join(data_mask, new_mask_name)
        


        prediction = output["sem_seg"].astype(np.uint8)

        pred_img = Image.fromarray(prediction, mode="P")
        pred_img.putpalette(np.array(PALETTE, dtype=np.uint8))

        results.append({
            "index": image_id,
            "img": img_path,
            "mask": np.array(pred_img, dtype=np.uint8),
            "difficulty": difficulty,
            "average_confidence": average_confidence,
            "coefficient": coefficient
        })

    df_results = pd.DataFrame(results)


    df_sorted = df_results.sort_values(by=["coefficient", "average_confidence"], ascending=False)

    output_csv_path = os.path.join(data_folder, "predictions.csv")
    df_sorted.to_csv(output_csv_path, index=False)

    print("Обработка завершена. Результаты сохранены в:", output_csv_path)