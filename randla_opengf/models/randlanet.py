import argparse
import json
import os
import os.path as osp
from typing import Optional, Union

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch import Tensor
from torch.nn import Linear, Sequential, Dropout
from torch_geometric.data import Data
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Linear, MessagePassing, knn_graph, radius_graph
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torchmetrics.functional.classification import jaccard_index

from models.base import SharedMLP, FPModule
from data import create_dataloader
from models import sampler
from models.sampler import create_sampler


class RandlaConv(MessagePassing):
    def __init__(self, d, d_slash):
        super(RandlaConv, self).__init__(aggr="add")
        self.mlp_rppe = SharedMLP([10, d])  # Relative Point Position Encoding
        self.mlp_att = Linear(2 * d, 2 * d, bias=False)
        self.mlp_post_pool = SharedMLP([2 * d, d_slash])
        self.d = d
        self.d_slash = d_slash

    def forward(self,
                x: Union[Tensor, PairTensor],
                pos: Union[Tensor, PairTensor],
                edge_index: Adj
                ) -> Tensor:

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)

        # propagate_type: (x: PairTensor, pos: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, size=None)
        out = self.mlp_post_pool(out)  # f_tilde
        return out

    def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor,
                index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        delta = pos_i - pos_j
        dist = delta.norm(dim=-1, keepdim=True)
        rp = torch.cat([pos_i, pos_j, delta, dist], dim=-1)
        r = self.mlp_rppe(rp)

        # makesure the dim of r and xj are both d and the size are the same
        assert x_j.shape == r.shape
        f_hat = torch.cat([x_j, r], dim=-1)

        # g(f,W) and do softmax along the KNN
        s_att = softmax(self.mlp_att(f_hat), index, ptr, size_i)
        # propagate will do sum aggregation after attentive pooling between KNN
        return s_att * f_hat

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.d}, {self.d_slash})'


class DilatedResidualBlock(torch.nn.Module):
    def __init__(self, d_in, d_out, k=16, r=-1):
        """
        最终输出特征维度是 2*d_out
        """
        super(DilatedResidualBlock, self).__init__()
        self.mlp_start = SharedMLP([d_in, d_out // 2])
        self.mlp_end = SharedMLP([d_out, 2 * d_out], act=None)  # end and skip are added and then activated
        self.mlp_skip = SharedMLP([d_in, 2 * d_out], act=None)

        self.locse_ap1 = RandlaConv(d_out // 2, d_out // 2)
        self.locse_ap2 = RandlaConv(d_out // 2, d_out)

        self.lrelu = torch.nn.LeakyReLU(0.2)
        self.k = k
        self.r = r


    def reset_parameters(self):
        reset(self.lrelu)

    def forward(self, x, pos, batch):
        if self.r > 0:
            edge_index = radius_graph(pos, r=self.r, batch=batch, loop=True, max_num_neighbors=self.k)
        else:
            edge_index = knn_graph(pos, k=self.k, batch=batch, loop=True)
        x_skip = self.mlp_skip(x)  # 2*dout

        x = self.mlp_start(x)  # dout / 2
        x = self.locse_ap1(x, pos, edge_index)  # dout / 2
        x = self.locse_ap2(x, pos, edge_index)  # dout
        x = self.mlp_end(x)  # 2*dout
        x = x + x_skip
        x = self.lrelu(x)  # 2 * dout
        return x


def inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv


class RandlaNet(torch.nn.Module):
    def __init__(self, config: DictConfig):
        super(RandlaNet, self).__init__()
        config = config.network
        k = config.k  # knn
        up_k = config.up_k
        self.decimations = OmegaConf.to_object(config.decimations)  # decimation ratio
        nb = config.num_bottleneck
        num_feats = config.num_features
        if config.append_height:
            num_feats += 1
        num_class = config.num_classes
        if config.ignore_class >= 0:
            num_class -= 1

        r = config.grid_size * 1.75
        self.voxel_size = config.grid_size
        self.ratio = 0.25
        self.sampler_method = config.sampler

        self.fc_start = SharedMLP([num_feats, nb])

        # encoder
        self.module_down = torch.nn.ModuleList([
            DilatedResidualBlock(nb, 2 * nb, k, r),  # 32
            DilatedResidualBlock(4 * nb, 8 * nb, k, 2 * r),  # 128
            DilatedResidualBlock(16 * nb, 16 * nb, k, 4 * r),  # 256
            DilatedResidualBlock(32 * nb, 32 * nb, k, 8 * r),  # 512
            DilatedResidualBlock(64 * nb, 64 * nb, k, 16 * r),  # 1024
        ])

        self.mlp_summit = SharedMLP([128 * nb, 128 * nb])

        self.module_up = torch.nn.ModuleList([
            FPModule(up_k, SharedMLP([128 * nb + 128 * nb, 64 * nb])),
            FPModule(up_k, SharedMLP([64 * nb + 64 * nb, 32 * nb])),
            FPModule(up_k, SharedMLP([32 * nb + 32 * nb, 16 * nb])),
            FPModule(up_k, SharedMLP([16 * nb + 16 * nb, 4 * nb])),
            FPModule(up_k, SharedMLP([4 * nb + 4 * nb, nb]))
        ])
        self.mlp_cls = Sequential(
            SharedMLP([nb, 64]),
            SharedMLP([64, 32]),
            Dropout(),
            SharedMLP([32, num_class], plain_last=True, act=None, norm=None)  # cls 节点是 softmax 激活
        )

    def forward(self, data: Data):
        x, pos, batch, ptr = data.x, data.pos, data.batch, torch.clone(data.ptr)

        # create random permutation for each batch
        B = ptr.size()[0] - 1

        # batch_sizes = ptr[1:] - ptr[:-1]
        #
        # indices = [torch.randperm(batch_sizes[i], dtype=torch.int64, device=ptr.device) + ptr[i] for i in range(B)]
        # indices = torch.cat(indices)
        # inverse_indices = inverse_permutation(indices)
        #
        # x = x[indices, :]
        # pos = pos[indices, :]
        # batch = batch[indices]

        out_x = []
        out_pos = []
        out_batch = []

        x = self.fc_start(x)

        for m, encoder in enumerate(self.module_down):
            # KNN aggregation
            x = encoder(x, pos, batch)
            out_x.append(x)
            out_pos.append(pos)
            out_batch.append(batch)

            # select first part of ::decimated indices in each batch
            # batch_sizes = torch.div(batch_sizes, self.decimations[m], rounding_mode='floor') + 1
            # indices_decimated = torch.cat([
            #     torch.arange(batch_sizes[i], device=ptr.device) + ptr[i] for i in range(B)
            # ])

            # update ptr
            # for i in range(B):
            #     ptr[i + 1] = ptr[i] + batch_sizes[i]
            '''
            采用fps采样方法，返回x， pos， batch
            '''
            idx_dst, batch_dst, pos_dst = create_sampler(pos, batch, self.ratio, self.sampler_method)
            x, pos, batch = x[idx_dst, :], pos_dst, batch_dst

        x = self.mlp_summit(x)

        for i, decoder in enumerate(self.module_up):
            skip_x = out_x.pop()
            skip_pos = out_pos.pop()
            skip_batch = out_batch.pop()

            x, pos, batch = decoder(x, pos, batch, skip_x, skip_pos, skip_batch)

        x = self.mlp_cls(x)
        # x = x[inverse_indices, :]
        return x

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='randla')
    parser.add_argument('--config', dest='config', default='../conf/randlanet.yaml',
                        help='configuration file in yaml format')
    parser.add_argument('--data', default=r'../conf/opengf.yaml')
    parser.add_argument('--train', default=r'../conf/train.yaml')
    args = parser.parse_args()
    config = os.path.abspath(args.config)
    cfg = OmegaConf.load(config)
    # data
    datacfg = osp.abspath(args.data)
    datacfg = OmegaConf.load(datacfg)
    # data
    # train
    traincfg = osp.abspath(args.train)
    traincfg = OmegaConf.load(traincfg)
    # train
    cfg = OmegaConf.merge(cfg, datacfg, traincfg)
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    cfg.train.checkpoints = osp.abspath(cfg.train.checkpoints)
    cfg.train.loggers = osp.abspath(cfg.train.loggers)
    train_loader = create_dataloader(cfg, 'train')
    test_loader = create_dataloader(cfg, 'val')


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    model = RandlaNet(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    with open(os.path.join(cfg.data.root, 'processed-0.20', 'metadata.json'), 'r') as f:
        _cla_weights = torch.tensor(json.load(f)['class_weights']).to(device)
        max_sampler = _cla_weights.max()
        _cla_weights = max_sampler / _cla_weights
        if cfg.data.ignore_class >= 0:
            _cla_weights = _cla_weights[[i for i in range(cfg.data.num_classes) if i != cfg.data.ignore_class]]
            # _cla_weights[cfg.data.ignore_class] = 0.0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=_cla_weights)


    def train():
        model.train()

        total_loss = correct_nodes = total_nodes = 0
        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            # out = F.softmax(out, dim=1)
            # loss = F.cross_entropy(out, data.y)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
            total_nodes += data.num_nodes

            if (i + 1) % 10 == 0:
                print(f'[{i + 1}/{len(train_loader)}] Loss: {total_loss / 10:.4f} '
                      f'Train Acc: {correct_nodes / total_nodes:.4f}')
                total_loss = correct_nodes = total_nodes = 0


    @torch.no_grad()
    def test(loader):
        model.eval()

        ious, categories = [], []
        y_map = torch.empty(loader.dataset.num_classes, device=device).long()
        for data in loader:
            data = data.to(device)
            sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()

            outs = model(data)
            for out, y, category in zip(outs.split(sizes), data.y.split(sizes),
                                        data.category.tolist()):
                category = list(ShapeNet.seg_classes.keys())[category]
                part = ShapeNet.seg_classes[category]
                part = torch.tensor(part, device=device)

                y_map[part] = torch.arange(part.size(0), device=device)

                iou = jaccard_index(out[:, part].argmax(dim=-1), y_map[y],
                                    num_classes=part.size(0), absent_score=1.0)
                ious.append(iou)

            categories.append(data.category)

        iou = torch.tensor(ious, device=device)
        category = torch.cat(categories, dim=0)

        mean_iou = scatter(iou, category, reduce='mean')  # Per-category IoU.
        return float(mean_iou.mean())  # Global IoU.


    for epoch in range(1, 3):
        train()
        # `iou = test(test_loader)
        # print(f'Epoch: {epoch:02d}, Test IoU: {iou:.4f}')`
        scheduler.step()