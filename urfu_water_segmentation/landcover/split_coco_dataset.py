#!/usr/bin/env python
import os
import os.path as osp
import json
import random

def split_coco_json(coco_file, out_train, out_val, val_fraction=0.2, seed=42):
    random.seed(seed)

    with open(coco_file, 'r') as f:
        dataset = json.load(f)

    images = dataset.get('images', [])
    annotations = dataset.get('annotations', [])
    categories = dataset.get('categories', [])
    info = dataset.get('info', {})
    licenses = dataset.get('license', [])


    random.shuffle(images)
    num_val = int(len(images) * val_fraction)


    val_images = images[:num_val]
    train_images = images[num_val:]


    val_img_ids = set(img['id'] for img in val_images)
    train_img_ids = set(img['id'] for img in train_images)


    train_annotations = []
    val_annotations = []
    for ann in annotations:
        if ann == "":
            break
        if ann['image_id'] in val_img_ids:
            val_annotations.append(ann)
        else:
            train_annotations.append(ann)


    train_coco = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories,
        "info": info,
        "license": licenses
    }

    val_coco = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": categories,
        "info": info,
        "license": licenses
    }




    os.makedirs(osp.dirname(out_train), exist_ok=True)
    with open(out_train, 'w') as f:
        json.dump(train_coco, f, indent=2)

    os.makedirs(osp.dirname(out_val), exist_ok=True)
    with open(out_val, 'w') as f:
        json.dump(val_coco, f, indent=2)

    print(f"Train set: {len(train_images)} images, {len(train_annotations)} annotations => {out_train}")
    print(f"Val set:   {len(val_images)} images, {len(val_annotations)} annotations => {out_val}")


if __name__ == '__main__':

    coco_file = "dataset_coco.json"                  
    out_train = "./train_coco.json"                    
    out_val = "./val_coco.json"                        


    split_coco_json(coco_file, out_train, out_val, val_fraction=0.2)
