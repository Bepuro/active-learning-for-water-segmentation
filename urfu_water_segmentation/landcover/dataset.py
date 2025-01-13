import os.path as osp
import json

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class LandcoverAI(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'object'),
        palette=[[0, 0, 0], [255, 255, 255]]
    )

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.png',
                 ann_file=None,
                 **kwargs):
        super().__init__(
            ann_file=ann_file,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs
        )

    
    def load_data_list(self):
        if not self.ann_file:
            return []
        print( self.ann_file)


        if "val" in self.ann_file:
            json_path = "./val_coco.json"
        else:
            json_path = "." + self.ann_file.split("/.")[1]
        if not osp.isfile(json_path):
            raise FileNotFoundError(f"COCO-JSON file not found: {json_path}")

        with open(json_path, 'r') as f:
            dataset_coco = json.load(f)

        images = dataset_coco.get('images', [])
        annotations = dataset_coco.get('annotations', [])

        imgid_to_info = {}
        for img_info in images:
            imgid = img_info['id']
            file_name = img_info['file_name']
            width = img_info.get('width', None)
            height = img_info.get('height', None)
            imgid_to_info[imgid] = (file_name, width, height)


        imgid_to_mask = {}
        for ann in annotations:
            image_id = ann['image_id']
            seg_map_file = ann['seg_map_file']

            imgid_to_mask[image_id] = seg_map_file

        data_list = []
        for img_id, (file_name, w, h) in imgid_to_info.items():
            seg_map_file = imgid_to_mask.get(img_id, None)
            if seg_map_file is None:
                continue
            
            img_path_abs = osp.join(self.data_root, file_name)
            seg_path_abs = osp.join(self.data_root, seg_map_file)

            data_info = dict(
                img_path=img_path_abs,
                seg_map_path=seg_path_abs,
            )
            if w is not None and h is not None:
                data_info['width'] = w
                data_info['height'] = h
            
            data_list.append(data_info)

        return data_list
    
    def get_data_info(self, idx):
        data_info = super().get_data_info(idx)
        return data_info
