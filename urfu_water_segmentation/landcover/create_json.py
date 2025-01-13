#!/usr/bin/env python

import os
import json
import glob
from PIL import Image  # pip install Pillow

def convert_landcover_to_json(data_root, out_file,
                              img_ext='tif', mask_ext='png'):
    images_dir = os.path.join(data_root, 'images')
    masks_dir = os.path.join(data_root, 'gt')


    image_paths = sorted(glob.glob(os.path.join(images_dir, f'*.{img_ext}')))

    images_list = []
    annotations_list = []

    for idx, img_path in enumerate(image_paths):
        print("1")
        file_name = os.path.basename(img_path)
        base_name, _ = os.path.splitext(file_name)


        mask_file_name = f"{base_name}.{mask_ext}"
        mask_path_rel = os.path.join("gt", mask_file_name)
        mask_path_full = os.path.join(masks_dir, mask_file_name)
        
        if mask_path_full == "":
            break

        if not os.path.isfile(mask_path_full):
            print(f"Warning: cannot find mask {mask_path_full} for image {img_path}, skipping it.")
            continue


        with Image.open(img_path) as pil_img:
            width, height = pil_img.size


        images_list.append({
            "id": idx,
            "file_name": os.path.join("images", file_name),
            "width": width,
            "height": height
        })


        annotations_list.append({
            "id": idx,
            "image_id": idx,
            "seg_map_file": mask_path_rel
        })


    categories_list = [
        {"id": 1, "name": "background"},
        {"id": 2, "name": "object"}
    ]

    dataset_dict = {
        "images": images_list,
        "annotations": annotations_list,
        "categories": categories_list
    }

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(dataset_dict, f, indent=2)

    print(f"Saved {len(images_list)} items into {out_file}")

if __name__ == '__main__':

    data_root = "/misc/home1/m_imm_freedata/Segmentation/Projects/mmseg_water/landcover.ai_512"
    out_file = "./dataset_coco.json"
    convert_landcover_to_json(data_root, out_file, img_ext='tif', mask_ext='png')
