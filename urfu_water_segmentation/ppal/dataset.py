from collections import defaultdict
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
import csv, mmcv, torch
import numpy as np
from mmseg.registry import DATASETS


DATASET_COLORMAP = dict(
    background=(0, 0, 0),
    water=(64, 64, 64),
    tree=(128, 128, 128),
    roads=(192, 192, 192),
    buildings=(255, 255, 255),
)

DATASET_CLASS_MAPPING = {
    'tree': 'background',
    'roads': 'background',
    'buildings': 'background',
}


@DATASETS.register_module()
class LandcoverAI(BaseSegDataset):
    METAINFO = dict(
        classes=list(DATASET_COLORMAP.keys()),
        palette=list(DATASET_COLORMAP.values()),
    )

    def __init__(self, **kwargs):
        new_classes = []
        self._data_label_map = {}
        
        for key, value in DATASET_COLORMAP.items():
            new_key = DATASET_CLASS_MAPPING.get(key, key)
            
            if new_key not in new_classes:
                new_classes.append(new_key)
            
            self._data_label_map[value[0]] = new_classes.index(new_key)
            
        super().__init__(
            img_suffix=".tif",
            seg_map_suffix=".tif",
            metainfo={'classes': new_classes},
            **kwargs
        )
        
    def get_data_info(self, idx):
        result = super().get_data_info(idx)
        if result is None:
            return None
        
        result['label_map'] = self._data_label_map.copy()
        return result
    
@DATASETS.register_module()
class ALDataset(LandcoverAI):

    def load_data_list(self):
        if not self.ann_file:
            raise ValueError('`ann_file` для LandcoverAI_CSV не задан')

        data_list = []
        with open(self.ann_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                info = dict(
                    img_path=row['img_path'],
                    seg_map_path=row['gt_path'],
                    reduce_zero_label=self.reduce_zero_label,
                    seg_fields=[]              
                )
                data_list.append(info)
        return data_list
    
    @staticmethod
    def load_img(path: str) -> torch.Tensor:
        img = mmcv.imread(path, flag='color')[..., ::-1].copy()
        #         img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        # return img
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    
    @staticmethod
    def load_mask(mask_path: str) -> torch.Tensor:
        # m = mmcv.imread(mask_path, flag='grayscale')      
        # m = (m == 64).astype(np.uint8)                    
        # return torch.from_numpy(m).long()
        m = mmcv.imread(mask_path, flag='grayscale')
        return torch.from_numpy((m == 64).astype(np.uint8)).long()
    
