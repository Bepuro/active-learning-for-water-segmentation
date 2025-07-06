# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
from pathlib import Path


from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from preparation import preparation
from active_learning_dcus import get_prediction
from active_learning_dbs import get_diverse_candidates
from emulate_data_formation import update_labeled, update_checkpoint,clean_mask_folder

from mmseg.registry import RUNNERS

# We must import dataset to register it
from dataset import ALDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    DATA_DIR = Path("data/glh_water/cache/")
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    tr_ds = cfg.train_dataloader.dataset
    tr_ds.type = 'ALDataset'
    tr_ds.data_root = ''
    tr_ds.ann_file = str((DATA_DIR / 'labeled.csv').resolve())
    print(tr_ds.ann_file)
    tr_ds.data_prefix = dict(img_path='', seg_map_path='')
    for split in ('val_dataloader', 'test_dataloader'):
        ds = getattr(cfg, split).dataset
        ds.type = 'LandcoverAI'
        ds.pop('ann_file', None)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # for key in ("default_hooks", "visualizer", "vis_backends"):
    #     cfg.pop(key, None)

    
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    preparation()

    for i in range(15):

        print("НАЧАЛО ОБУЧЕНИЯ")

        main()
        print("ВЫЧИСЛЯЕМ МАСКУ и неуверенность")
        get_prediction()
        print("оТБИРАЕМ КАНДИДАТОВ")
        get_diverse_candidates()

        update_labeled()
        update_checkpoint()
        #clean_mask_folder()
