from typing import Tuple

import torch
import torch_scatter
from torch import Tensor
from torch_geometric.nn import fps, voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster


def fps_sampler(pos: Tensor, batch: Tensor, ratio: float = 0.25) -> Tuple[Tensor, Tensor, Tensor]:
    r"""采用 FPS 算法进行采样

    Args:
        pos (Tensor): 原始点云位置信息
            :math:`x \in \mathbb{R}^{N \times 3}`.

        batch (Tensor): 对应 batch 信息的索引
            :math:`\mathbf{b} \in {\{ 0,0,0,\ldots,B-1\}}^N`

        grid_size (float): 对于 FPS 降采样不需要，但是为了保持接口哦一致性保留

        ratio (float): 降采样比率，必须小于 1

    Returns:
        idx_dst (Tensor): 返回采样的索引号

        batch_dst (Tensor): 返回采样的 batch 号，保证每个 batch 至少有一个点

        pos_dst (Tensor): 返回采样的位置，对于 FPS 直接用索引号获取

    :rtype: :class:`Tuple[Tensor, Tensor, Tensor]`
    """

    idx_dst = fps(pos, batch, ratio)
    batch_dst = batch[idx_dst]
    pos_dst = pos[idx_dst, :]
    return idx_dst, batch_dst, pos_dst


def random_sampler(pos: Tensor, batch: Tensor, grid_size: float, ratio: float = 0.25) -> Tuple[Tensor, Tensor, Tensor]:
    r"""采用随机索引降采样

    Args:
        pos (Tensor): 原始点云位置信息
            :math:`x \in \mathbb{R}^{N \times 3}`.

        batch (Tensor): 对应 batch 信息的索引
            :math:`\mathbf{b} \in \mathbb{Z}^N {\{ 0,0,0,\ldots,B-1\}}`

        grid_size (float): 降采样的网格大小，随机采样用不上

        ratio (float): 降采样比率

    Returns:
        idx_dst (Tensor): 返回采样的索引号

        batch_dst (Tensor): 返回采样的 batch 号，保证每个 batch 至少有一个点

        pos_dst (Tensor): 返回采样的位置，直接根据索引号获取

    :rtype: :class:`Tuple[Tensor, Tensor, Tensor]`
    """
    # 首先计算 ptr，虽然这个可以作为输入，但是保证接口简单性，计算也不复杂
    assert pos.size(0) == batch.numel()
    batch_size = int(batch.max()) + 1

    deg = pos.new_zeros(batch_size, dtype=torch.long)
    deg.scatter_add_(0, batch, torch.ones_like(batch))

    ptr = deg.new_zeros(batch_size + 1)
    torch.cumsum(deg, 0, out=ptr[1:])

    count = ptr[1:] - ptr[:-1]
    decim_count = torch.div(count, round(1.0 / ratio), rounding_mode='floor')
    decim_count.clamp_(min=1)  # Prevent empty examples.

    decim_indices = [
        ptr[i] + torch.randperm(count[i], device=ptr.device)[:decim_count[i]]
        for i in range(batch_size)
    ]
    decim_indices = torch.cat(decim_indices, dim=0)

    return decim_indices, batch[decim_indices], pos[decim_indices]


def voxel_sampler(pos: Tensor, batch: Tensor, grid_size: float, ratio: float = 0.25) -> Tuple[Tensor, Tensor, Tensor]:
    r"""采用网格降采样

    Args:
        pos (Tensor): 原始点云位置信息
            :math:`x \in \mathbb{R}^{N \times 3}`.

        batch (Tensor): 对应 batch 信息的索引
            :math:`\mathbf{b} \in \mathbb{Z}^N {\{ 0,0,0,\ldots,B-1\}}`

        grid_size (float): 降采样的网格大小，通常是搜索半径的 1/2.5

        ratio (float): 降采样比率，网格降采样用不上

    Returns:
        idx_dst (Tensor): 返回采样的索引号

        batch_dst (Tensor): 返回采样的 batch 号，保证每个 batch 至少有一个点

        pos_dst (Tensor): 返回采样的位置，是网格中的平均位置

    :rtype: :class:`Tuple[Tensor, Tensor, Tensor]`
    """

    cluster = voxel_grid(pos, grid_size, batch)
    cluster, perm = consecutive_cluster(cluster)

    pos_dst = torch_scatter.scatter(pos, cluster, dim=0, reduce='mean')
    batch_dst = batch[perm]
    return perm, batch_dst, pos_dst


def create_sampler(pos: Tensor, batch: Tensor, ratio: float = 0.25,
            method: str = 'fps') -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    创建采样接口函数，
    接口输入是：

    pos (Tensor): 原始点云位置信息
        :math:`x \in \mathbb{R}^{N \times 3}`.

    batch (Tensor): 对应 batch 信息的索引
        :math:`\mathbf{b} \in \mathbb{Z}^N {\{ 0,0,0,\ldots,B-1\}}`

    grid_size (float): 降采样的网格大小，通常是搜索半径的 1/2.5，fps 和 random 用不上

    ratio (float): 降采样比率，网格降采样用不上

    接口输出是：

    idx_dst (Tensor): 返回采样的索引号

    batch_dst (Tensor): 返回采样的 batch 号，保证每个 batch 至少有一个点

    pos_dst (Tensor): 返回采样的位置，是网格中的平均位置
    Args:
        method (str): 采样方法，可以是 fps, random, voxel

    Returns:
        返回接口函数
        (pos: Tensor, batch: Tensor, grid_size: float, ratio: float) -> Tuple[Tensor, Tensor, Tensor]

    """
    if method == 'fps':
        return fps_sampler(pos, batch, ratio)
    elif method == 'random':
        return random_sampler
    else:
        return voxel_sampler
