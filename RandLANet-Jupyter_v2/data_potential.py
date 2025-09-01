import pickle
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Union, List, Tuple
from utils.tools import DataProcessing as DP
from sklearn.neighbors import KDTree
import random
from torchvision import transforms
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少日志输出

class RandomScale(object):
    def __init__(self, device=torch.device('cuda'), scale_min=0.8, scale_max=1.2):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.device = device

    def __call__(self, xyz: torch.Tensor):
        scale_factor = torch.tensor(
            [random.uniform(self.scale_min, self.scale_max)],
            device=self.device
        )
        return xyz * scale_factor


class RandomRotate(object):
    def __init__(self, device=torch.device('cuda'), rotation_range=(0, 360)):
        self.rot_min, self.rot_max = rotation_range
        self.device = device

    def __call__(self, xyz: torch.Tensor):
        angle = torch.deg2rad(
            torch.tensor([random.uniform(self.rot_min, self.rot_max)], device=self.device)
        )
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        rot_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], device=xyz.device).float()
        return torch.matmul(xyz, rot_matrix.T)


class JitterPoints(object):
    def __init__(self, device=torch.device('cuda'), sigma=0.01, clip=0.03):
        self.sigma = sigma
        self.clip = clip
        self.device = device

    def __call__(self, xyz: torch.Tensor):
        noise = torch.normal(0, self.sigma, size=xyz.shape, device=self.device)
        noise = torch.clamp(noise, -self.clip, self.clip)
        return xyz + noise

# 创建组合变换
def get_transforms( mode='train'):
    """根据模式获取适当的变换组合"""
    device = torch.device('cpu')
    if mode == 'train':
        return transforms.Compose([
            RandomRotate(device, rotation_range=(0, 360)),
            RandomScale(device, scale_min=0.8, scale_max=1.2),
            JitterPoints(device, sigma=0.01, clip=0.03),
        ])
    elif mode == 'val':
        return None  # 验证集无需数据增强
    else:  # test
        return None


class CloudsDataset(Dataset):
    def __init__(self, config, split="train", transform=False):
        self.num_steps = None
        self.num_files = None
        self.config = config
        self.split = split
        self.transform = get_transforms(split) if transform is True else None

        self.metajson_path = os.path.join(self.processed_dir, 'metadata.json')
        with open(self.metajson_path, 'r') as f:
            self.splits = json.load(f)

        frenquency = torch.tensor(self.splits['class_weights'])
        frenquency = frenquency / torch.sum(frenquency)
        self._class_weights = torch.sqrt(frenquency)

        # save all the Data
        self.inputs: List[np] = []
        # save for each cloud a kdtree index
        self.kdtrees: List[KDTree] = []

        self.load_data()

    def __len__(self):
        """返回数据集中的点云数量"""
        return self.num_steps

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['arradata.json']

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.config["data_root"], f'processed-{self.config["grid_size"]:.2f}')

    def __getitem__(self, idx):
        """获取单个点云样本"""
        while True:
            # Get potential points from tree structure
            cloud_data: np = self.inputs[idx]
            kdtree: KDTree = self.kdtrees[idx]

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

            num_points = cloud_data.shape[0]
            point_idx = random.randint(0, num_points - 1)
            # Center point of input region
            center = pos[point_idx, :].reshape(1, -1).astype(np.float32)
            center += np.random.rand(1, 3) * 0.5

            _, indices = kdtree.query(center, k=self.config["num_points_per_step"])

            indices = indices[0]

            if indices.shape[0] >= self.config["num_points_per_step"]:
                # 点数太少重新选择点
                break

        indices = DP.shuffle_idx(indices)
        data_out = {}

        pos = pos[indices, :] - center
        xyz_tensor = torch.tensor(pos, dtype=torch.float32, device='cpu')  # 指定 device

        if self.transform:
            xyz_tensor = self.transform(xyz_tensor)

        data_out['xyz'] = xyz_tensor
        if intensity is not None:
            intensity = intensity[indices]
            intensity_tensor = torch.tensor(intensity, dtype=torch.float32, device='cpu')
            data_out['intensity'] = intensity_tensor
        if rgb is not None:
            rgb = rgb[indices, :]
            rgb_tensor = torch.tensor(rgb, dtype=torch.float32, device='cpu') / 255.0  # 归一化到 0-1
            data_out['rgb'] = rgb_tensor
        label = label[indices]
        labels_tensor = torch.tensor(label, dtype=torch.int32, device='cpu')
        data_out['class'] = labels_tensor
        data_out['index'] = torch.tensor(indices, dtype=torch.int32, device='cpu')

        return data_out

    def load_data(self):
        files = self.splits[self.split]
        self.num_files = len(files)

        self.num_steps = self.num_files

        for file in files:
            npy_path = os.path.join(self.processed_dir, file + '.npy')
            kdt_path = os.path.join(self.processed_dir, file + '_KDTree.pkl')

            with open(kdt_path, 'rb') as f:
                kdtree = pickle.load(f)
            data = np.load(npy_path, allow_pickle=False)

            self.inputs.append(data)
            self.kdtrees.append(kdtree)


class ActiveLearningSampler(IterableDataset):
    def __init__(self, config, dataset: CloudsDataset, batch_size=8, step_size=4, split='train', transform=False):
        self.config = config
        self.dataset = dataset
        self.n_steps = step_size
        self.split = split
        self.batch_size = batch_size
        self.possibility = {}
        self.min_possibility = {}
        self.transform = get_transforms(split) if transform is True else None

        #Random initialisation for weights
        self.possibility[split] = []
        self.min_possibility[split] = []
        for i, points in enumerate(self.dataset.inputs):
            self.possibility[split] += [np.random.rand(points['x'].shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

    def __iter__(self):
        return self.spatially_regular_gen()

    def __len__(self):
        return self.n_steps # not equal to the actual size of the dataset, but enable nice progress bars

    def spatially_regular_gen(self):
        # Choosing the least known point as center of a new cloud each time.
        for i in range(self.n_steps * self.batch_size):  # num_per_epoch
            # t0 = time.time()
            # Choose a random cloud
            cloud_idx = int(np.argmin(self.min_possibility[self.split]))

            # choose the point with the minimum of possibility as query point
            point_ind = np.argmin(self.possibility[self.split][cloud_idx])

            # Get points from tree structure
            points = np.concatenate([
                np.expand_dims(self.dataset.inputs[cloud_idx]['x'], 1),
                np.expand_dims(self.dataset.inputs[cloud_idx]['y'], 1),
                np.expand_dims(self.dataset.inputs[cloud_idx]['z'], 1)
            ], axis=-1).astype(np.float32)

            label = self.dataset.inputs[cloud_idx]['class']

            rgb = None
            intensity = None

            if self.config["HasRGB"]:
                rgb = np.concatenate([
                    np.expand_dims(self.dataset.inputs[cloud_idx]['r'], 1),
                    np.expand_dims(self.dataset.inputs[cloud_idx]['g'], 1),
                    np.expand_dims(self.dataset.inputs[cloud_idx]['b'], 1)
                ], axis=-1)

            if self.config["HasIntensity"]:
                intensity = self.dataset.inputs[cloud_idx]['intensity']

            # Center point of input region
            center_point = points[point_ind, :].reshape(1, -1)

            # Add noise to the center point
            noise = np.random.normal(scale=3.5 / 10, size=center_point.shape)
            pick_point = center_point + noise.astype(center_point.dtype)

            if len(points) < self.config["num_points_per_step"]:
                _, queried_idx = self.dataset.kdtrees[cloud_idx].query(pick_point, k=len(points))
            else:
                _, queried_idx = self.dataset.kdtrees[cloud_idx].query(pick_point, k=self.config["num_points_per_step"])

            queried_idx = queried_idx[0]

            queried_idx = DP.shuffle_idx(queried_idx)
            # Collect points and colors
            queried_pc_xyz = points[queried_idx]
            queried_pc_xyz = queried_pc_xyz - pick_point

            data_out = {}

            xyz_tensor = torch.tensor(queried_pc_xyz, dtype=torch.float32, device='cpu')  # 指定 device

            if self.transform:
                xyz_tensor = self.transform(xyz_tensor)

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
            self.possibility[self.split][cloud_idx][queried_idx] += delta
            self.min_possibility[self.split][cloud_idx] = float(np.min(self.possibility[self.split][cloud_idx]))

            yield data_out
