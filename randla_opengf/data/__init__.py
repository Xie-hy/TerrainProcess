import copy
import json
import os
from typing import Optional, List

import torch.utils.data
import torch_geometric.transforms as T
from filelock import FileLock
from omegaconf import DictConfig
from torch.multiprocessing import Lock
# from torch_geometric.data import LightningDataset, Dataset
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader

from data.transform import RandomDropColor, AutoColorContrast, AppendHeight
from data.opengf import OpenGF

global_lock = Lock()


def worker_init_fn_filelock(workerid):
    worker_info = torch.utils.data.get_worker_info()
    dir = worker_info.dataset.processed_dir
    split = worker_info.dataset.split
    seed_everything(os.getpid())
    worker_info.dataset.lock = FileLock(f'{dir}_{split}.lock')

def worker_init_fn_lock(workerid):
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.lock = global_lock

def get_class_names(config: DictConfig) -> List[str]:
    ignore = config.data.ignore_class
    if config.data.name == 'opengf':
        class_name = OpenGF.label_to_names
    else:
        raise NotImplementedError(f'This type {config.data.name} is not implemented')
    class_name = list(class_name.values())
    # 删除unlabelled
    class_name = class_name.pop(ignore) if ignore >= 0 else class_name
    return class_name



def create_dataloader(cfg: DictConfig, split='train'):
    if split == 'train':
        transforms = T.Compose([
            T.Center(),
            T.RandomFlip(0)
        ])

        if 'random_scale_offset' in cfg.data.transform:
            transforms.transforms.append(
                T.RandomScale((1 - cfg.data.transform.random_scale_offset, 1 + cfg.data.transform.random_scale_offset)))
        if 'random_translate' in cfg.data.transform:
            transforms.transforms.append(T.RandomJitter(cfg.data.transform.random_translate))
        if 'random_rotate_z' in cfg.data.transform and ('use_profile' not in cfg.data or not cfg.data.use_profile):
            transforms.transforms.append(T.RandomRotate(cfg.data.transform.random_rotate_z, 2))
        if 'append_height' in cfg.data.transform and cfg.data.transform.append_height:
            transforms.transforms.append(AppendHeight())
        if 'random_drop_color' in cfg.data.transform:
            transforms.transforms.append(RandomDropColor(cfg.data.transform.random_drop_color, cfg.data.num_features))

        batch_size = cfg.train.batch_size
    else:
        transforms = T.Compose([T.Center()])
        if 'append_height' in cfg.data.transform and cfg.data.transform.append_height:
            transforms.transforms.append(AppendHeight())
        batch_size = cfg.train.batch_size_val

    if cfg.data.name == 'opengf':
        dataset = OpenGF(cfg, split=split, transform=transforms)
    else:
        raise NotImplementedError('The dataset is not implementd')

    print(f'create dataloader {split} in pid {os.getpid()}')

    return DataLoader(dataset, batch_size=batch_size, num_workers=cfg.train.num_workers, pin_memory=True, shuffle=False,
                      persistent_workers=True, worker_init_fn=worker_init_fn_filelock)
