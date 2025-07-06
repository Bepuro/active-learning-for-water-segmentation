import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import ast


def parse_mask(s):
    s = s.replace('\n', ' ').replace('[', '').replace(']', '')
    arr = np.fromstring(s, sep=' ', dtype=np.uint8)
    return arr


def compute_iou(mask1, mask2):
    """
    Вычисляет IoU для двух бинарных масок.
    Предполагается, что маски – это массивы, где значения >128 интерпретируются как 1 (объект),
    а остальные – как 0 (фон).
    """
    bin_mask1 = (mask1 > 128).astype(np.uint8)
    bin_mask2 = (mask2 > 128).astype(np.uint8)
    intersection = np.logical_and(bin_mask1, bin_mask2).sum()
    union = np.logical_or(bin_mask1, bin_mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def compute_histogram_distance(img1, img2):
    """
    Вычисляет расстояние между гистограммами двух изображений.
    Здесь используется корреляция (чем ниже корреляция, тем больше расстояние).
    Приводим гистограммы к нормированному виду.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    
    corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)    
    d = 1 - (corr + 1) / 2
    return d

# --------------------------
# 1. Загрузка кандидатов и выбор нижних 40%
# --------------------------

def get_diverse_candidates():
    candidate_csv = "./data/glh_water/cache/predictions.csv" 
    df_candidates = pd.read_csv(candidate_csv)

    all_data_csv = "./data/glh_water/unlabeled.csv"
    df_all_data = pd.read_csv(all_data_csv)
    all_len = len(df_all_data)


    #n_bottom = int(0.4 * n_total)
    n_bottom = all_len
    if n_bottom < 600:
        n_bottom = 600
    df_bottom = df_candidates.tail(n_bottom).reset_index(drop=True)


    # --------------------------
    # 2. Вычисление попарного сходства
    # --------------------------
    mask_paths = df_bottom['mask'].apply(parse_mask).tolist()
    img_paths = df_bottom['img']

    n = len(mask_paths)

    mask_distance_matrix = np.zeros((n, n))
    img_distance_matrix = np.zeros((n, n))

    print("Вычисление попарных расстояний между масками и изображениями...")



    masks =mask_paths
    imgs = [None] * len(mask_paths)


    # for i in range(n):
    #     if not os.path.exists(mask_paths[i]):
    #         print(f"Маска не найдена: {mask_paths[i]}")
    #         continue
    #     masks[i] = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)
    #     if not os.path.exists(img_paths[i]):
    #         print(f"Изображение не найдено: {img_paths[i]}")
    #         continue
    #     imgs[i] = cv2.imread(img_paths[i], cv2.IMREAD_COLOR)


    for i in range(n):
        imgs[i] = cv2.imread(img_paths[i], cv2.IMREAD_COLOR)
    for i in range(n):
        for j in range(i+1, n):
            if masks[i] is None or masks[j] is None:
                continue

            
            
            iou = compute_iou(masks[i], masks[j])
            d_mask = 1 - iou
            mask_distance_matrix[i, j] = d_mask
            mask_distance_matrix[j, i] = d_mask
            
            if imgs[i] is None or imgs[j] is None:
                continue
            d_img = compute_histogram_distance(imgs[i], imgs[j])
            img_distance_matrix[i, j] = d_img
            img_distance_matrix[j, i] = d_img

    w_mask = 0.7  
    w_img  = 0.3  
    distance_matrix = w_mask * mask_distance_matrix + w_img * img_distance_matrix

    # --------------------------
    # 3. Этап 1: k-Center-Greedy
    # --------------------------
    desired_count =  n_bottom // 5  
    selected_indices = []

    selected_indices.append(0)
    remaining = set(range(n)) - set(selected_indices)

    while len(selected_indices) < desired_count and remaining:
        max_min_distance = -1
        candidate_to_add = None
        for idx in remaining:
            distances = [distance_matrix[idx, sel] for sel in selected_indices]
            min_distance = min(distances)
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                candidate_to_add = idx
        if candidate_to_add is not None:
            selected_indices.append(candidate_to_add)
            remaining.remove(candidate_to_add)
        else:
            break

    print("После k-Center-Greedy выбраны индексы:", selected_indices)

    # --------------------------
    # 4. Этап 2: Модифицированный k-Means++
    # --------------------------
    initial_centers = selected_indices.copy()
    cluster_assignment = {}

    for idx in range(n):
        distances_to_centers = [distance_matrix[idx, center] for center in initial_centers]
        assigned_cluster = np.argmin(distances_to_centers)
        cluster_assignment.setdefault(assigned_cluster, []).append(idx)

    final_selected = []
    for cluster_idx, indices in cluster_assignment.items():
        if len(indices) == 0:
            continue
        sum_distances = {}
        for i in indices:
            d_sum = sum([distance_matrix[i, j] for j in indices])
            sum_distances[i] = d_sum
        rep_idx = min(sum_distances, key=sum_distances.get)
        final_selected.append(rep_idx)

    print("После модифицированного k-Means++ выбраны индексы:", final_selected)

    # --------------------------
    # 5. Итоговый DataFrame с выбранными разнообразными изображениями
    # --------------------------
    df_diverse = df_bottom.iloc[final_selected].reset_index(drop=True)
    print("Итоговый набор разнообразных кандидатов:")
    print(df_diverse)

    output_csv = "./data/glh_water/cache/diverse_candidates.csv"
    df_diverse.to_csv(output_csv, index=False)
    print(f"Результаты diversity-based sampling сохранены в: {output_csv}")