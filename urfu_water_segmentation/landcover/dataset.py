import os.path as osp
import json

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class LandcoverAI(BaseSegDataset):
    # meta-инфа, если у вас есть несколько классов
    METAINFO = dict(
        classes=('background', 'object'),
        palette=[[0, 0, 0], [255, 255, 255]]
    )

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.png',  # если маски .png
                 ann_file=None,
                 **kwargs):
        super().__init__(
            ann_file=ann_file,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs
        )
        # можно здесь задать свои поля, если нужно
        # self.some_param = ...
    
    def load_data_list(self):
        if not self.ann_file:
            # Если ann_file не указан, возможно вернём пустой список (или ошибку).
            return []
        print("1")
        json_path = osp.join(self.data_root, self.ann_file)  # data_root + ann_file
        if not osp.isfile(json_path):
            raise FileNotFoundError(f"COCO-JSON file not found: {json_path}")

        with open(json_path, 'r') as f:
            dataset_coco = json.load(f)

        images = dataset_coco.get('images', [])
        annotations = dataset_coco.get('annotations', [])

        # Для быстрого доступа: image_id -> (file_name, width, height)
        imgid_to_info = {}
        for img_info in images:
            imgid = img_info['id']
            file_name = img_info['file_name']  # например, "images/00001.tif"
            width = img_info.get('width', None)
            height = img_info.get('height', None)
            imgid_to_info[imgid] = (file_name, width, height)

        # Аналогично: image_id -> seg_map_file (или список, если несколько аннотаций)
        # Но чаще в MMSeg каждый image_id имеет одну маску. 
        # В упрощённом примере предполагаем 1:1
        imgid_to_mask = {}
        for ann in annotations:
            image_id = ann['image_id']
            seg_map_file = ann['seg_map_file']
            # Если хотите поддержать несколько аннотаций на 1 изображение —
            # нужно аккумулировать список.
            imgid_to_mask[image_id] = seg_map_file

        data_list = []
        for img_id, (file_name, w, h) in imgid_to_info.items():
            seg_map_file = imgid_to_mask.get(img_id, None)
            if seg_map_file is None:
                # если нет аннотации для этого image_id, пропускаем или записываем как unlabeled
                continue
            
            # Превращаем относительный путь в абсолютный (если нужно)
            img_path_abs = osp.join(self.data_root, file_name)
            seg_path_abs = osp.join(self.data_root, seg_map_file)

            # Собираем элемент
            data_info = dict(
                img_path=img_path_abs,
                seg_map_path=seg_path_abs,
            )
            # Можно указать размеры
            if w is not None and h is not None:
                data_info['width'] = w
                data_info['height'] = h
            
            data_list.append(data_info)

        return data_list
    
    def get_data_info(self, idx):
        data_info = super().get_data_info(idx)
        return data_info
