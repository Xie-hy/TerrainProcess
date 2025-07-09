import os # noqa
import json
import pickle
import warnings
from typing import Union, List, Tuple

import numpy as np
import pdal
import torch
from omegaconf import DictConfig
from sklearn.neighbors import KDTree
from torch import LongTensor
from torch_geometric.data import Data

from data.cloud_dataset import PointcloudDataset

# ignore user warning
warnings.filterwarnings("ignore", category=UserWarning)


class OpenGF(PointcloudDataset):
    """
    OpenGF: An Ultra-Large-Scale Ground Filtering Dataset Built Upon Open ALS Point Clouds Around the World
    https://github.com/Nathan-UW/OpenGF

    1: Unclassified
    2: Ground

    """

    label_to_names = {
        0: 'unclassified',
        1: 'ground'
    }


    def __init__(self, config: DictConfig, split='train', transform=None, pre_transform=None):
        super(OpenGF, self).__init__(config, split, transform, pre_transform)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        names = []
        for path, subdirs, files in os.walk(self.raw_dir):
            for name in files:
                filepath = os.path.join(path, name)
                relpath = os.path.relpath(filepath, self.raw_dir)
                names.append(relpath)

        return names

    def convert_label_origin_to_train(self, indices: LongTensor) -> LongTensor:
        return indices

    def convert_label_train_to_origin(self, indices: LongTensor) -> LongTensor:
        return indices

    def process(self):
        if os.path.exists(self.processed_paths[0]):
            return

        class_counts = None
        train = []
        test = []
        val = []

        # 1. 首先每个文件都生成一个 json 的索引，统计 intensity 最大最小值
        for raw_path in self.raw_paths:
            name = os.path.splitext(os.path.basename(raw_path))[0]

            if 'Train' in raw_path:
                train.append(name)
            elif 'Test' in raw_path:
                test.append(name)
            elif 'Validation' in raw_path:
                val.append(name)

            raw_path = raw_path.replace(os.sep, '/')
            kdt_path = os.path.join(self.processed_dir, name + '.pkl')
            pth_path = os.path.join(self.processed_dir, name + '.pt')

            if os.path.exists(pth_path):
                # 所有文件都有标记信息
                data = torch.load(pth_path)
                _, counts = torch.unique(data.y, return_counts=True)
                class_counts = class_counts + counts if class_counts is not None else counts
                continue

            pipeline = pdal.Pipeline()
            pipeline |= pdal.Reader.las(filename=raw_path)
            pipeline |= pdal.Filter.stats(dimensions='Intensity')
            pipeline |= pdal.Filter.voxelcenternearestneighbor(cell=self.config.grid_size)
            #outlier剔除场景中过高或过低的异常点
            pipeline |= pdal.Filter.outlier(method='statistical',
                                            multiplier=3.0)
            pipeline |= pdal.Filter.range(limits='Classification[1:2]')
            pipeline |= pdal.Filter.assign(value=["Classification = 0 WHERE Classification == 1",
                                                  "Classification = 1 WHERE Classification == 2"])

            if self.has_dimensionality:
                # Linearity, Planarity, Scattering, Verticality
                pipeline |= pdal.Filter.covariancefeatures(threads=8,
                                                           feature_set='Dimensionality')

            count = pipeline.execute()
            arrays = pipeline.arrays[0]
            metadata = pipeline.metadata['metadata']
            stats = metadata['filters.stats']['statistic'][0]

            bounds = metadata['readers.las']
            minx = bounds['minx']
            miny = bounds['miny']
            minz = bounds['minz']

            pos = np.concatenate([
                np.expand_dims(arrays['X'] - minx, 1),
                np.expand_dims(arrays['Y'] - miny, 1),
                np.expand_dims(arrays['Z'] - minz, 1)
            ], axis=-1).astype(np.float32)
            y = arrays['Classification']

            x = None
            if x is None:
                x = np.zeros([pos.shape[0], 1], dtype=np.float32)

            if self.has_intensity:
                stats = metadata['filters.stats']['statistic'][0]
                intensity = self.normalize_attributes(arrays['Intensity'], 0.0, (float)(stats['maximum']),
                                                      stats['average'],
                                                      stats['stddev'])
                x = intensity
            if self.has_dimensionality:
                dimensionality = np.concatenate([
                    np.expand_dims(arrays['Linearity'], 1),
                    np.expand_dims(arrays['Planarity'], 1),
                    np.expand_dims(arrays['Scattering'], 1),
                    np.expand_dims(arrays['Verticality'], 1)
                ], axis=-1).astype(np.float32)
                if x is None:
                    x = dimensionality
                else:
                    x = np.concatenate([x, dimensionality], axis=-1)

            kdtree = KDTree(pos)
            with open(kdt_path, 'wb') as f:
                pickle.dump(kdtree, f)

                # x为特征包括intensity或者intensity + dimensionality
                # y为分类标签
                # pos为xyz坐标
            data = Data(x=torch.tensor(x), y=torch.tensor(y).long(), pos=torch.tensor(pos),
                        offset=torch.tensor([minx, miny, minz]))
            torch.save(data, pth_path)

            # 如果class_counts不是None，class_counts = class_counts + counts;
            # 如果class_counts是None，class_counts = counts;
            _, counts = torch.unique(data.y, return_counts=True)
            class_counts = class_counts + counts if class_counts is not None else counts

        splits = {
            'train': train,
            'val': val,
            'test': test,
            'class_weights': class_counts.tolist(),
        }
        with open(self.processed_paths[0], 'w') as f:
            json.dump(splits, f, indent=4)


if __name__ == '__main__':
    from omegaconf import OmegaConf
    import argparse
    from multiprocessing import Lock
    import os


    def main(cfg):
        OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
        cfg.data.root = os.path.abspath(cfg.data.root)
        dataset = OpenGF(cfg, split='train')
        dataset.lock = Lock()


    parser = argparse.ArgumentParser(description='OpenGF dataset')
    parser.add_argument('--config', dest='config', default='../conf/opengf.yaml',
                        help='configuration file in yaml format')

    args = parser.parse_args()
    config = os.path.abspath(args.config)
    cfg = OmegaConf.load(config)
    main(cfg)
