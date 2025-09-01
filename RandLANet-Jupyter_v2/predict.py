import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
import pdal
from sklearn.neighbors import KDTree
from typing import Union, Any
import random
import os


def normalize_attributes(array: np.array, min_value: float, max_value: float, mean_value: float, std_value: float):
    min_value = np.clip(mean_value - 2 * std_value, min_value, max_value)
    max_value = np.clip(mean_value + 2 * std_value, min_value, max_value)
    value = (array.astype(np.float32) - min_value) / (max_value - min_value)
    value = np.clip(value, 0.0, 1.0)
    return np.expand_dims(value, 1)


class PredictCloudDataset(Dataset):
    def __init__(self, config):
        self.center = None
        self.config = config
        self.path = self.config["input_path"]

        # save all the Data
        self.inputs: np = None
        self.raw_pos: np = None
        # save for each cloud a kdtree index
        self.kdtree: Union[Any, KDTree] = None
        self.num_files = None

        self.load_data()

    def __len__(self):
        """返回数据集中的点云数量"""
        return self.num_files

    def __getitem__(self, idx):
        """获取单个点云样本"""
        cloud_data: np = self.inputs
        kdtree: KDTree = self.kdtree

        pos = np.concatenate([
            np.expand_dims(cloud_data['x'], 1),
            np.expand_dims(cloud_data['y'], 1),
            np.expand_dims(cloud_data['z'], 1)
        ], axis=-1).astype(np.float32)

        label = cloud_data['class']

        rgb = None
        intensity = None

        if self.config["HasRGB"]:
            rgb = np.concatenate([
                np.expand_dims(cloud_data['r'], 1),
                np.expand_dims(cloud_data['g'], 1),
                np.expand_dims(cloud_data['b'], 1)
            ], axis=-1)

        if self.config["HasIntensity"]:
            intensity = cloud_data['intensity']

        num_points = pos.shape[0]
        point_idx = random.randint(0, num_points - 1)
        # Center point of input region
        center = pos[point_idx, :].reshape(1, -1).astype(np.float32)
        center += np.random.rand(1, 3) * 0.5

        _, indices = kdtree.query(center, k=self.config["num_points_per_step"])

        indices = indices[0]

        data_out = {}

        pos = pos[indices, :] - center
        xyz_tensor = torch.tensor(pos, dtype=torch.float32)
        data_out['xyz'] = xyz_tensor
        if intensity is not None:
            intensity = intensity[indices]
            intensity_tensor = torch.tensor(intensity, dtype=torch.float32)
            data_out['intensity'] = intensity_tensor
        if rgb is not None:
            rgb = rgb[indices, :]
            rgb_tensor = torch.tensor(rgb, dtype=torch.float32) / 255.0  # 归一化到 0-1
            data_out['rgb'] = rgb_tensor
        label = label[indices]
        labels_tensor = torch.tensor(label, dtype=torch.int32)
        data_out['class'] = labels_tensor
        data_out['index'] = torch.tensor(indices, dtype=torch.int32)

        return data_out

    def load_data(self):
        """
        Load all pytorch data into memory
        """
        self.num_files = self.config["step_size"]
        extension_name = os.path.splitext(self.path)[1].lower()
        assert extension_name == '.las' or extension_name == '.laz'

        pipeline = pdal.Pipeline()
        pipeline |= pdal.Reader.las(filename=self.path)
        pipeline |= pdal.Filter.stats(dimensions="Intensity")  # 统计
        pipeline |= pdal.Filter.voxelcenternearestneighbor(cell=self.config["grid_size"])

        pipeline.execute()

        metadata = pipeline.metadata['metadata']

        arrays = pipeline.arrays[0]

        bounds = metadata['readers.las']
        minx = bounds['minx']
        miny = bounds['miny']
        minz = bounds['minz']
        self.center = torch.tensor([minx, miny, minz], dtype=torch.float64)

        self.raw_pos = np.concatenate([
            np.expand_dims(arrays['X'], 1),
            np.expand_dims(arrays['Y'], 1),
            np.expand_dims(arrays['Z'], 1)
        ], axis=-1).astype(np.float64)

        pos = np.concatenate([
            np.expand_dims(arrays['X'] - minx, 1),
            np.expand_dims(arrays['Y'] - miny, 1),
            np.expand_dims(arrays['Z'] - minz, 1)
        ], axis=-1).astype(np.float32)

        point_npy = np.zeros(pos.shape[0], dtype=[
            ('x', np.float32),  # X坐标 (浮点数)
            ('y', np.float32),  # Y坐标 (浮点数)
            ('z', np.float32),  # Z坐标 (浮点数)
            ('intensity', np.float32),  # 强度值 (浮点数)
            ('r', np.uint16),  # 红色分量 (0-255)
            ('g', np.uint16),  # 绿色分量 (0-255)
            ('b', np.uint16),  # 蓝色分量 (0-255)
            ('class', np.uint8)  # 类别分量 (int)
        ])

        if self.config["PredictHasRGB"]:
            red = arrays['Red']
            green = arrays['Green']
            blue = arrays['Blue']
            point_npy['r'] = red
            point_npy['g'] = green
            point_npy['b'] = blue

        label = arrays['Classification']
        if label is None:
            label = np.zeros(pos.shape[0])

        point_npy['x'] = (arrays['X'] - minx).astype(np.float32)
        point_npy['y'] = (arrays['Y'] - miny).astype(np.float32)
        point_npy['z'] = (arrays['Z'] - minz).astype(np.float32)
        point_npy['class'] = label

        if self.config["PredictHasIntensity"]:
            stats = metadata['filters.stats']['statistic'][0]
            intensity = normalize_attributes(arrays['Intensity'], 0.0, (float)(stats['maximum']),
                                             stats['average'],
                                             stats['stddev'])
            point_npy['intensity'] = intensity.reshape(intensity.shape[0])

        self.kdtree = KDTree(pos)
        self.inputs = point_npy


class ActiveLearningSampler(IterableDataset):
    def __init__(self, config, dataset: PredictCloudDataset, batch_size=8, step_size=3):
        self.config = config
        self.dataset = dataset
        self.batch_size = batch_size
        self.possibility = {}
        self.min_possibility = {}

        self.n_steps = step_size

        #Random initialisation for weights
        self.possibility = np.random.rand(self.dataset.inputs['x'].shape[0]) * 1e-3

    def __iter__(self):
        return self.spatially_regular_gen()

    def __len__(self):
        return self.n_steps # not equal to the actual size of the dataset, but enable nice progress bars

    def spatially_regular_gen(self):
        # Choosing the least known point as center of a new cloud each time.
        for i in range(self.n_steps * self.batch_size):  # num_per_epoch

            # choose the point with the minimum of possibility as query point
            point_ind = np.argmin(self.possibility)

            # Get points from tree structure
            points = np.concatenate([
                np.expand_dims(self.dataset.inputs['x'], 1),
                np.expand_dims(self.dataset.inputs['y'], 1),
                np.expand_dims(self.dataset.inputs['z'], 1)
            ], axis=-1).astype(np.float32)

            label = self.dataset.inputs['class']

            rgb = None
            intensity = None

            if self.config["HasRGB"]:
                rgb = np.concatenate([
                    np.expand_dims(self.dataset.inputs['r'], 1),
                    np.expand_dims(self.dataset.inputs['g'], 1),
                    np.expand_dims(self.dataset.inputs['b'], 1)
                ], axis=-1)

            if self.config["HasIntensity"]:
                intensity = self.dataset.inputs['intensity']

            # Center point of input region
            center_point = points[point_ind, :].reshape(1, -1)

            # Add noise to the center point
            noise = np.random.normal(scale=3.5 / 10, size=center_point.shape)
            pick_point = center_point + noise.astype(center_point.dtype)

            if len(points) < self.config["num_points_per_step"]:
                _, queried_idx = self.dataset.kdtree.query(pick_point, k=len(points))
            else:
                _, queried_idx = self.dataset.kdtree.query(pick_point, k=self.config["num_points_per_step"])

            queried_idx = queried_idx[0]

            # Collect points and colors
            queried_pc_xyz = points[queried_idx]
            queried_pc_xyz = queried_pc_xyz - pick_point

            data_out = {}

            xyz_tensor = torch.tensor(queried_pc_xyz, dtype=torch.float32, device='cpu')  # 指定 device

            data_out['xyz'] = xyz_tensor
            if intensity is not None:
                intensity = intensity[queried_idx]
                intensity_tensor = torch.tensor(intensity, dtype=torch.float32, device='cpu')
                data_out['intensity'] = intensity_tensor
            if rgb is not None:
                rgb = rgb[queried_idx, :]
                rgb_tensor = torch.tensor(rgb, dtype=torch.float32, device='cpu') / 255.0  # 归一化到 0-1
                data_out['rgb'] = rgb_tensor

            label = label[queried_idx]
            labels_tensor = torch.tensor(label, dtype=torch.int32, device='cpu')
            data_out['class'] = labels_tensor
            data_out['index'] = torch.tensor(queried_idx, dtype=torch.int32, device='cpu')

            dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
            delta = np.square(1 - dists / np.max(dists))

            self.possibility[queried_idx] += delta  # 此时支持数组索

            yield data_out
