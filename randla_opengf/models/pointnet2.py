import os

import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.nn import MLP, PointNetConv, fps, knn_interpolate, radius_graph


class SAModule(torch.nn.Module):
    def __init__(self, local_nn, global_nn=None, ratio=0.25, r=0.1):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(local_nn, global_nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        indices = fps(pos, batch, self.ratio)

        edge_index = radius_graph(pos, self.r, batch=batch)
        x = self.conv(x, pos, edge_index)

        x_dst = x[indices]
        pos_dst = pos[indices]
        batch_dst = batch[indices]

        return x, x_dst, pos_dst, batch_dst


class FPModule(torch.nn.Module):
    def __init__(self, nn, k):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class PointNetNetwork(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.network
        self.down_modules = torch.nn.ModuleList()
        self.up_modules = torch.nn.ModuleList()
        # fix
        r = config.data.grid_size * 2.5
        # fix

        self.summit_mlp = MLP(OmegaConf.to_object(self.config.summit_layers), norm=None, plain_last=False)
        for i, layers in enumerate(self.config.down_mlp_layers):
            layers = OmegaConf.to_object(layers)
            if i == 0:
                layers[0] = self.config.num_features + 1 if self.config.append_height else self.config.num_features
            layers[0] += 3
            self.down_modules.append(
                SAModule(MLP(layers, plain_last=False), ratio=self.config.ratios[i], r=r * 2 ** i))
        for i, layers in enumerate(self.config.up_mlp_layers):
            layers = OmegaConf.to_object(layers)
            self.up_modules.append(FPModule(MLP(layers, plain_last=False), self.config.up_k[i]))
        cls_mlp_layers = OmegaConf.to_object(self.config.cls_mlp_layers)
        cls_mlp_layers.append(self.config.num_classes)
        self.cls = MLP(cls_mlp_layers, norm=None)

    def forward(self, data: Data):
        skip_data = []

        x = data.x
        pos = data.pos
        batch = data.batch

        for module in self.down_modules:
            x, x_sa, pos_sa, batch_sa = module(x, pos, batch)
            skip_data.append((x, pos, batch))

            x = x_sa
            pos = pos_sa
            batch = batch_sa

        x = self.summit_mlp(x)

        for i, module in enumerate(self.up_modules):
            skip_x, skip_pos, skip_batch = skip_data.pop()
            x, pos, batch = module(x, pos, batch, skip_x, skip_pos, skip_batch)

        return self.cls(x)


if __name__ == '__main__':
    from zero3d.data.semantic3d import Semantic3dDataset
    import torch_geometric.transforms as T
    from torch_geometric.loader import DataLoader
    import torch.nn.functional as F
    import argparse
    from torch.multiprocessing import Lock


    def main(cfg):
        OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly

        cfg.data.root = os.path.abspath(cfg.data.root)

        transforms = T.Compose([
            T.RandomJitter(cfg.data.transform.random_translate),
            T.RandomRotate(cfg.data.transform.random_rotate_z, 2)
        ])

        dataset = Semantic3dDataset(cfg, split='train', transform=transforms)
        dataset.lock = Lock()
        loader = DataLoader(dataset, batch_size=6, shuffle=False, num_workers=2)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PointNetNetwork(cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        def train():
            model.train()

            total_loss = correct_nodes = total_nodes = 0
            for i, data in enumerate(loader):
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = F.cross_entropy(out, data.y, ignore_index=cfg.data.ignore_class)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
                total_nodes += data.num_nodes

                if (i + 1) % 10 == 0:
                    print(f'[{i + 1}/{len(loader)}] Loss: {total_loss / 10:.4f} '
                          f'Train Acc: {correct_nodes / total_nodes:.4f}')
                    total_loss = correct_nodes = total_nodes = 0

        for epoch in range(1, 31):
            train()
            scheduler.step()


    parser = argparse.ArgumentParser(description='Pointnet2')
    parser.add_argument('--config', dest='config', default='./conf/pointnet2.yaml',
                        help='configuration file in yaml format')
    args = parser.parse_args()
    config = os.path.abspath(args.config)
    cfg = OmegaConf.load(config)
    main(cfg)
