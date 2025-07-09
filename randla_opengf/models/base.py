import torch
from torch.nn import Sequential
from torch_geometric.nn import MLP, knn_interpolate, voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_batch
from torch_scatter import scatter


class SharedMLP(MLP):
    """SharedMLP with new defauts BN and Activation following tensorflow implementation."""
    lrelu_kwargs = {"negative_slope": 0.2}
    bn_kwargs = {"momentum": 0.01, "eps": 1e-6}

    def __init__(self, *args, **kwargs):
        #  v
        kwargs["plain_last"] = False
        # LeakyRelu with 0.2 slope by default.
        kwargs["act"] = kwargs.get("act", "LeakyReLU")
        kwargs["act_kwargs"] = kwargs.get("act_kwargs", self.lrelu_kwargs)
        # BatchNorm with 0.99 momentum and 1e-6 eps by defaut.
        kwargs["norm_kwargs"] = kwargs.get("norm_kwargs", self.bn_kwargs)
        super().__init__(*args, **kwargs)


class FPModule(torch.nn.Module):
    """Upsampling with a skip connection."""

    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class SAModule(torch.nn.Module):
    """
    Downsampling with grid sampling and maxpooling
    """

    def __init__(self, grid_size):
        super(SAModule, self).__init__()
        self.grid_size = grid_size

    def forward(self, x, pos, batch):
        # 1. 利用 voxel grid 降采样
        cluster = voxel_grid(pos=pos, size=self.grid_size, batch=batch)
        cluster, perm = consecutive_cluster(cluster)

        x_dst = scatter(x, cluster, dim=0, reduce='max')
        pos_dst = scatter(pos, cluster, dim=0, reduce='mean')
        batch_dst = pool_batch(perm, batch)

        return x_dst, pos_dst, batch_dst

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.grid_size})'


class MultiInputSequential(Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input
