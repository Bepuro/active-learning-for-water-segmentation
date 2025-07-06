import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


_DATA_ROOT = Path('/misc/home1/m_imm_freedata/Segmentation')
DATASETS = {'landcover':_DATA_ROOT / 'Projects/mmseg_water/landcover.ai_512/train',
            'glh_water': _DATA_ROOT / 'Projects/GLH_water/glh_cut_512_filtered/train',
            'deepglobe':_DATA_ROOT / 'DeepGlobe_Land/DeepGlobe512/train',
            'loveda':_DATA_ROOT / 'LoveDA/train'}

mask_path = "gt/"
image_path = "images/"

for dataset_name, dataset_path in DATASETS.items():
    base_path = str(dataset_path)
    mask_folder = os.path.join(base_path, mask_path)
    image_folder = os.path.join(base_path, image_path)

    mask_files = sorted(os.listdir(mask_folder))
    image_files = sorted(os.listdir(image_folder))

    data = []
    for mask_file, image_file in zip(mask_files, image_files):
        data.append({
                'gt_path': os.path.join(mask_folder, mask_file),
                'img_path': os.path.join(image_folder, image_file)
            })
    df = pd.DataFrame(data)
    df.reset_index(inplace=True)

    df = df.copy()
    df['gt_path_clean'] = df['gt_path'].str.replace(r'\.(png|tif)$', '', regex=True)
    df['img_path_clean'] = df['img_path'].str.replace(r'\.(png|tif)$', '', regex=True)
    df['gt_path_clean'] = df['gt_path_clean'].str.replace(os.path.join(base_path, "gt/"), '', regex=False)
    df['img_path_clean'] = df['img_path_clean'].str.replace(os.path.join(base_path, "images/"), '', regex=False)
    df['same'] = (df['gt_path_clean'] == df['img_path_clean']).astype(int)
    result_df = df.drop(['gt_path_clean', 'img_path_clean', 'same'], axis=1)

    if not os.path.exists("data"):
        os.makedirs("data")
    os.makedirs("data"+"/"+dataset_name, exist_ok=True)
    result_df.to_csv(os.path.join("data"+"/"+dataset_name, "all_data.csv"), index=False)

    annotator, labeled = train_test_split(result_df, test_size=0.2, random_state=42)

    labeled.to_csv(os.path.join("data"+"/"+dataset_name, "labeled.csv"), index=False)
    annotator.to_csv(os.path.join("data"+"/"+dataset_name, "annotator.csv"), index=False)
    unlabaled_set = annotator.drop(['gt_path'], axis=1)
    unlabaled_set.to_csv(os.path.join("data"+"/"+dataset_name, "unlabeled.csv"), index=False)








