import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss


def knn_faiss(points: torch.Tensor, k: int, device: torch.device, exclude_self=True):
    """
    设备感知的KNN计算，自动处理CPU/GPU设备切换
    Args:
        points: (B, N, 3) 输入点云坐标
        k: 邻居数量
        device: 目标设备（自动选择CPU/GPU实现）
        exclude_self: 是否排除自身点
    """
    B, N, D = points.shape
    k_val = min(k, N - 1) if exclude_self else min(k, N)

    # 结果预分配
    all_indices = torch.empty(B, N, k_val, dtype=torch.long, device=device)
    all_distances = torch.empty(B, N, k_val, dtype=torch.float32, device=device)

    # 按设备选择实现
    use_gpu = device.type == 'cuda'
    if use_gpu:
        res = faiss.StandardGpuResources()  # GPU资源管理器
        index_type = faiss.GpuIndexFlatL2(res, D)  # GPU索引
    else:
        index_type = faiss.IndexFlatL2(D)  # CPU索引

    # 逐批次处理
    for i in range(B):
        batch_points = points[i].contiguous()
        batch_points_cpu = batch_points.cpu().numpy()  # 转换为CPU numpy数组

        index_type.reset()
        index_type.add(batch_points_cpu)  # 添加当前批次的数据

        # 查询k+1保证排除自身后有足够邻居
        search_k = k_val + 1 if exclude_self else k_val
        dist, idx = index_type.search(batch_points_cpu, search_k)

        if exclude_self:
            all_indices[i] = torch.from_numpy(idx[:, 1:]).to(device)
            all_distances[i] = torch.from_numpy(dist[:, 1:]).to(device)
        else:
            all_indices[i] = torch.from_numpy(idx).to(device)
            all_distances[i] = torch.from_numpy(dist).to(device)

    return all_indices, all_distances


class SharedMLP(nn.Module):
    """固定结构的通道适配MLP，避免动态创建层"""

    def __init__(self, in_channels, out_channels, bn=True, activation=nn.ReLU()):
        super().__init__()
        self.adapter = nn.Conv1d(in_channels, in_channels, 1) if in_channels != in_channels else nn.Identity()
        self.conv = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels) if bn else nn.Identity()
        self.activation = activation

    def forward(self, x):
        x = self.adapter(x)
        if x.dim() == 4:  # [B, C, N, K]
            B, C, N, K = x.shape
            x = x.reshape(B, C, N * K)  # 压缩为 [B, C, N*K]（3D）
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x) if self.activation else x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors, device):
        super().__init__()
        self.mlp = SharedMLP(10, d, activation=nn.ReLU())  # 输入通道=10
        self.device = device

    def forward(self, coords, features, knn_output):
        idx, dist = knn_output
        idx = idx.to(self.device)
        dist = dist.to(self.device)
        B, N, K = idx.shape

        # 1. 邻居坐标聚合 (B,3,N,K)
        base_coords = coords.transpose(1, 2).unsqueeze(-1)  # (B,3,N,1)
        idx_exp = idx.view(B, 1, N, K).expand(-1, 3, -1, -1)  # (B,3,N,K)
        neighbor_coords = torch.gather(
            coords.unsqueeze(2).expand(B, N, K, 3),
            1,
            idx_exp.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)  # (B,3,N,K)

        # 2. 计算空间特征：相对位置+距离
        rel_pos = base_coords - neighbor_coords  # (B,3,N,K)
        dist_tensor = dist.unsqueeze(1)  # (B,1,N,K)
        spatial_features = torch.cat([
            base_coords.expand_as(neighbor_coords),  # 广播至(B,3,N,K)
            neighbor_coords,
            rel_pos,
            dist_tensor
        ], dim=1)  # (B, 10, N, K)

        # 3. 压缩维度适配SharedMLP
        spatial_features = spatial_features.reshape(B, 10, N * K)
        spatial_feat = self.mlp(spatial_features)  # (B, d, N*K)
        spatial_feat = spatial_feat.reshape(B, -1, N, K)  # (B, d, N, K)

        # 4. 融合原始特征
        if features.dim() == 3:
            features = features.unsqueeze(-1)  # (B, C, N, 1)
        features_exp = features.expand(-1, -1, -1, K)  # (B, C, N, K)

        return torch.cat([spatial_feat, features_exp], dim=1)


class AttentivePooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, in_channels)  # 输入输出均为C
        self.softmax = nn.Softmax(dim=-2)  # 在K维度做softmax

    def forward(self, x):
        B, C, N, K = x.shape
        # 将通道维度移到最后，合并B和N维度
        x_perm = x.permute(0, 2, 1, 3)  # (B, N, C, K)
        x_flat = x_perm.reshape(B * N, C, K)  # (B*N, C, K)

        # 转置以适配线性层: (B*N, K, C)
        x_trans = x_flat.permute(0, 2, 1)  # (B*N, K, C)

        # 线性层处理
        attn_scores = self.linear(x_trans)  # (B*N, K, C)
        attn_scores = self.softmax(attn_scores)

        # 恢复原始维度
        scores = attn_scores.permute(0, 2, 1)  # (B*N, C, K)

        scores = scores.reshape(B, N, C, K)  # (B, N, C, K)

        scores = scores.permute(0, 2, 1, 3)  # (B, C, N, K)

        # 注意力加权池化
        pooled = torch.sum(scores * x, dim=-1, keepdim=True)  # (B, C, N, 1)
        return pooled


class LocalFeatureAggregation(nn.Module):
    """修复设备传递和KNN调用"""

    def __init__(self, d_in, d_out, num_neighbors, device):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.device = device

        self.mlp1 = SharedMLP(d_in, d_out // 2, activation=nn.LeakyReLU(0.2))
        self.lse1 = LocalSpatialEncoding(d_out // 2, num_neighbors, device)
        self.pool1 = AttentivePooling(d_out)
        self.mlp_pool1 = SharedMLP(d_out, d_out // 4)

        self.lse2 = LocalSpatialEncoding(d_out // 4, num_neighbors, device)
        self.pool2 = AttentivePooling(d_out // 2)
        self.mlp2 = SharedMLP(d_out // 2, d_out)

        # 残差分支（固定结构）
        self.residual = SharedMLP(d_in, d_out)
        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, features):
        B, _, N = features.shape[:3]

        # 关键修复：在当前下采样点云上计算KNN
        knn_output = knn_faiss(coords, self.num_neighbors, self.device)

        x = self.mlp1(features)
        x = self.lse1(coords, x, knn_output)
        x = self.mlp_pool1(self.pool1(x))
        x = self.lse2(coords, x, knn_output)
        x_main = self.mlp2(self.pool2(x))
        residual = self.residual(features)
        fused = torch.cat((x_main, residual), dim=1)

        return self.lrelu(fused)


class RandLANet(nn.Module):
    """核心改进：随机采样 + 设备一致性"""
    def __init__(self, d_in, num_classes, num_neighbors=16, decimation=4, device=torch.device('cuda')):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.decimation = decimation
        self.device = device

        # 初始特征提取
        self.fc_start = nn.Linear(d_in, 8)
        self.bn_start = nn.Sequential(
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2)
        )

        # 编码器
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(8, 16, num_neighbors, device),
            LocalFeatureAggregation(32, 64, num_neighbors, device),
            LocalFeatureAggregation(128, 128, num_neighbors, device),
            LocalFeatureAggregation(256, 256, num_neighbors, device)
        ])

        # 解码器
        self.decoder = nn.ModuleList([
            SharedMLP(1024, 256),
            SharedMLP(512, 128),
            SharedMLP(256, 32),
            SharedMLP(64, 8)
        ])

        # 分类头
        self.fc_end = nn.Sequential(
            SharedMLP(8, 64),
            SharedMLP(64, 32),
            nn.Dropout(0.5),
            SharedMLP(32, num_classes)
        )

    def random_sample(self, coords, features, target_num):
        """实现真正的随机采样（非切片）[citatation:2]"""
        idx = torch.randperm(coords.shape[1])[:target_num]
        return coords[:, idx], features[:, :, idx]

    def forward(self, input):
        # 自动对齐设备
        B, N, _ = input.shape
        coords = input[..., :3].to(self.device)
        input = input.to(self.device)

        # 初始特征
        x = self.bn_start(self.fc_start(input).transpose(1, 2))

        # 编码阶段
        x_stack = []
        num_points_stack = []
        decim_ratio = 1
        target_N = N
        for i, lfa in enumerate(self.encoder):
            # 特征聚合
            x = lfa(coords, x)
            x_stack.append(x.clone())
            num_points_stack.append(target_N)
            target_N = max(1, int(N / (decim_ratio * self.decimation)))

            coords, x = self.random_sample(coords, x, target_N)
            decim_ratio *= self.decimation

        # 解码阶段
        for i, mlp in enumerate(self.decoder):
            # 上采样目标点数
            target_Up_N = num_points_stack.pop()

            # 特征插值
            x = F.interpolate(x, size=target_Up_N, mode='nearest')
            coords = F.interpolate(coords.transpose(1, 2), size=target_Up_N, mode='nearest').transpose(1, 2)

            # 跳跃连接
            skip = x_stack.pop()
            if skip.shape[2] != target_N:
                skip = F.interpolate(skip, size=target_Up_N, mode='nearest')
            x = torch.cat([x, skip], dim=1)
            x = mlp(x)

        return self.fc_end(x).squeeze(-1)