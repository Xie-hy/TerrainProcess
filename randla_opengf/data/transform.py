import random

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('random_drop_color')
class RandomDropColor(BaseTransform):
    """Randomly sets color to zero
    """

    def __init__(self, p=0.2, num_featurs=3):
        super(RandomDropColor, self).__init__()
        self.p = p
        self.c = num_featurs

    def __call__(self, data: Data):
        if random.random() < self.p:
            data.x[:, :self.c] = 0
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'


@functional_transform('auto_color_contrast')
class AutoColorContrast(BaseTransform):
    def __init__(self, mean_value: Tensor, std_value: Tensor):
        """
        根据均值和标准差归一化颜色
        Args:
            mean_value: 范围为 [0-1]
            std_value: 范围为 [0-1]
        """
        super(AutoColorContrast, self).__init__()
        self.mean = mean_value.squeeze(0)
        self.std = std_value.squeeze(0)

    def __call__(self, data: Data):
        # 要确保这个在 append height 之前调用
        # x 要么是 rgb 颜色，要么是强度属性
        assert data.x.size(1) <= 3
        if data.x.max() > 1 and self.mean.max() > 1:
            data.x /= 255.
        data.x = (data.x - self.mean) / self.std
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(mean={self.mean.tolist()},std={self.std.tolist()})'


@functional_transform('append_height')
class AppendHeight(BaseTransform):
    def __init__(self):
        """
        把高程坐标加入特征，可以使特征对目标的绝对尺寸有概念
        """
        super(AppendHeight, self).__init__()

    def __call__(self, data: Data):
        z = data.pos[:, -1]
        z = (z - torch.min(z)).unsqueeze(-1)
        data.x = torch.cat([data.x, z], dim=-1)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
