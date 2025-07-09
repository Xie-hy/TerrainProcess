import json
import os
import pickle
from abc import ABC
from typing import Union, List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.neighbors import KDTree
from torch import Tensor, LongTensor
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import LinearTransformation
from shapely.geometry import LineString, Point, box

from data.dataprocessing import DataProcessing as DP


class PointcloudDataset(Dataset, ABC):
    # The lock must be set outside the initializer to be used in ddp_spawn
    lock = None

    # 在预处理部分ignore_class已经被删除
    def __init__(self, config: DictConfig, split="train", transform=None, pre_transform=None):
        self.config: DictConfig = config.data
        self.split = split

        super(PointcloudDataset, self).__init__(self.config.root, pre_transform=pre_transform, transform=transform)

        self._num_total_points = 0

        with open(self.processed_paths[0], 'r') as f:
            self.splits = json.load(f)

        frenquency = torch.tensor(self.splits['class_weights'])
        frenquency = frenquency / torch.sum(frenquency)
        self._class_weights = torch.sqrt(frenquency)

        # save all the Data
        self.inputs: List[Data] = []
        # save for each cloud a kdtree index
        self.kdtrees: List[KDTree] = []

        # save for each point a value to do balanced loading
        self.potentials: List[Tensor] = []
        # save for each cloud a value to record the min potentials
        self.min_potentials = []
        # save for each cloud a value to record the index of the min potentials
        self.argmin_potentials = []

        self.kdtrees2d: List[KDTree] = []
        self.bounds2d: List[Tensor] = []

        self.load_data()

        for i, data in enumerate(self.inputs):
            potentials = torch.randn((data.pos.shape[0],), dtype=torch.float64) * 1e-3
            self.potentials += [potentials]

            min_ind = int(torch.argmin(potentials))
            min_pot = float(potentials[min_ind])
            self.argmin_potentials += [min_ind]
            self.min_potentials += [min_pot]

        # Share potential memory
        self.argmin_potentials = torch.tensor(self.argmin_potentials, dtype=torch.int64)
        self.min_potentials = torch.tensor(self.min_potentials, dtype=torch.float64)
        self.argmin_potentials.share_memory_()
        self.min_potentials.share_memory_()
        for i in range(len(self.inputs)):
            self.potentials[i].share_memory_()

        # Test workers
        self.num = torch.zeros((1,)).long()
        self.num.share_memory_()

    @property
    def num_classes(self):
        """
        包含无效类别的总数目
        """
        return self.config.num_classes

    @property
    def num_features(self) -> int:
        return self.config.num_features

    @property
    def ignore_class(self) -> int:
        """
        原始的忽略类别
        """
        return self.config.ignore_class

    @property
    def has_dimensionality(self) -> bool:
        if self.config.num_features == 5 or self.config.num_features == 7:
            return True
        else:
            return False

    @property
    def has_intensity(self) -> bool:
        if self.config.HasIntensity:
            return True
        else:
            return False


    @property
    def class_weights(self) -> Tensor:
        return self._class_weights

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['metadata.json']

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, f'processed-{self.config.grid_size:.2f}')

    @property
    def num_total_points(self) -> int:
        return self._num_total_points

    @property
    def num_points_per_step(self) -> int:
        return self.config.num_points_per_step

    @staticmethod
    def normalize_attributes(array: np.array, min_value: float, max_value: float, mean_value: float, std_value: float):
        """根据均值和标准差进行归一化操作，采用2倍标准差归一化

        Args:
            array (np.array): 输入数据
            min_value (float): 初始最小值
            max_value (float): 初始最大值
            mean_value (float): 均值
            std_value (float): 标准差

        Returns:
            object (np.array): 归一化，并且维度变成2维，方便 cat

        """
        min_value = np.clip(mean_value - 2 * std_value, min_value, max_value)
        max_value = np.clip(mean_value + 2 * std_value, min_value, max_value)
        value = (array.astype(np.float32) - min_value) / (max_value - min_value)
        value = np.clip(value, 0.0, 1.0)
        return np.expand_dims(value, 1)

    def convert_label_origin_to_train(self, indices: LongTensor) -> LongTensor:
        """
        从原始标签，转换为忽略无效标签后的标签，例如如果0为无效标签，所有无效标签修改为-1,其他标签从0：C-1 映射成 0：C-2
        """
        raise NotImplementedError

    def convert_label_train_to_origin(self, indices: LongTensor) -> LongTensor:
        """
        从训练时的标签，转换为原始数据的标签
        """
        raise NotImplementedError

    def load_data(self):
        """
        Load all processed data into memory
        """

        files = self.splits[self.split]
        for file in files:
            # load all downsampled data
            pth_path = os.path.join(self.processed_dir, file + '.pt')
            kdt_path = os.path.join(self.processed_dir, file + '.pkl')
            kdt2d_path = os.path.join(self.processed_dir, file + '_xy.pkl')

            with open(kdt_path, 'rb') as f:
                kdtree = pickle.load(f)
            data = torch.load(pth_path, weights_only=False)

            if 'ALS' in self.config.keys() and self.config.ALS:
                xy = data.pos[:, :2]
                if os.path.exists(kdt2d_path):
                    with open(kdt2d_path, 'rb') as f:
                        kdtree2d = pickle.load(f)
                else:
                    kdtree2d = KDTree(xy.numpy())
                    with open(kdt2d_path, 'wb') as f:
                        pickle.dump(kdtree2d, f)
                self.kdtrees2d.append(kdtree2d)
                bounds2d = torch.cat([torch.min(xy, dim=0)[0], torch.max(xy, dim=0)[0]])
                self.bounds2d.append(bounds2d)

            self.inputs.append(data)
            self.kdtrees.append(kdtree)
            self._num_total_points += data.num_nodes

    def len(self) -> int:
        return self.config.num_steps_per_epochs

    def get(self, idx: int) -> Data:
        """
        Load all pytorch data into memory
        """

        return self.get_regular_neighbor()

    def get_regular_neighbor(self) -> Data:
        assert self.lock is not None
        while True:
            with self.lock:
                cloud_idx = int(torch.argmin(self.min_potentials))
                point_idx = int(self.argmin_potentials[cloud_idx])
                # self.num[0] += 1
                # print(self.num[0])

                # Get potential points from tree structure
                cloud_data: Data = self.inputs[cloud_idx]
                kdtree: KDTree = self.kdtrees[cloud_idx]

                # Center point of input region
                center = cloud_data.pos[point_idx, :].reshape(1, -1).numpy().astype(np.float32)
                center += np.random.rand(1, 3) * 0.5
                if self.config.radius > 0:
                    # 采用半径搜索
                    indices, dists = kdtree.query_radius(center, self.config.radius, return_distance=True,
                                                         sort_results=True)
                else:
                    # 采用 knn 搜索
                    dists, indices = kdtree.query(center, k=self.num_points_per_step)

                indices = indices[0]
                dists = dists[0]
                if indices.shape[0] > self.num_points_per_step:
                    indices = indices[:self.num_points_per_step]
                    dists = dists[:self.num_points_per_step]

                dists = np.square(dists)
                max_dist = dists[-1] + 0.01  # 防止为 0

                # 点越少，权重越小，下次继续选择该点的概率就大
                points_weights = cloud_data.y[indices]
                points_weights = self.class_weights.index_select(-1, points_weights)
                points_weights = points_weights.numpy()

                # 更新 potentials 的权重
                tukeys = np.square(1 - dists / max_dist) * points_weights

                self.potentials[cloud_idx][indices] += tukeys
                min_point_idx = torch.argmin(self.potentials[cloud_idx])
                self.min_potentials[[cloud_idx]] = self.potentials[cloud_idx][min_point_idx]
                self.argmin_potentials[[cloud_idx]] = min_point_idx

                if indices.shape[0] > 100:
                    # 点数太少重新选择点
                    break

        indices = DP.shuffle_idx(indices)

        pos = cloud_data.pos[indices, :] - torch.tensor(center)
        x = cloud_data.x[indices, :]
        y = self.convert_label_origin_to_train(cloud_data.y[indices]) if cloud_data.y is not None else None

        return Data(pos=pos, x=x, y=y)

