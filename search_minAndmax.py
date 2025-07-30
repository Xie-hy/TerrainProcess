import pdal
import laspy
import numpy as np
from sklearn.neighbors import NearestNeighbors

def knn_outlier_removal(distance, k=5, distance_threshold_factor=2.0):
    """
    使用 KNN 算法检测强度值离群点
    参数：
        distance: 强度值数组
        k: 近邻数量
        distance_threshold_factor: 距离阈值倍数（默认为 2 倍平均距离）
    返回：
        remove_mask: 布尔掩码，True 表示保留点，False 表示离群点
    """
    # 将强度值重塑为二维数组（样本数 x 特征数）
    distance_reshaped = distance.reshape(-1, 1)

    # 构建 KNN 模型
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(distance_reshaped)

    # 计算每个点的 K 近邻距离
    distances, _ = knn.kneighbors(distance_reshaped)

    # 计算每个点的平均距离
    avg_distances = np.mean(distances, axis=1)

    # 计算距离阈值（平均距离的倍数）
    threshold = np.mean(avg_distances) * distance_threshold_factor

    # 标记离群点（平均距离大于阈值）
    mask = avg_distances <= threshold

    return mask


# 连续颜色映射（推荐）
def distance_to_color(distances):

    min_val = np.nanmin(distances)
    max_val = np.nanmax(distances)
    print('max distance:', max_val)
    print('min distance:', min_val)
    normalized = (distances - min_val) / (max_val - min_val)
    # 生成颜色（从绿色到红色）
    red = normalized
    green = 1 - normalized
    blue = np.zeros_like(normalized)
    colors = np.stack([red, green, blue], axis=-1)

    return colors


pipeline = pdal.Pipeline()
pipeline |= pdal.Reader.las(filename='C:/Users/xhy/Desktop/pos1_pos3_difference.las')
# 执行PDAL管道
pipeline.execute()
arrays = pipeline.arrays[0]

dist = arrays['M3C2 distance']

points = np.concatenate([
        np.expand_dims(arrays['X'], 1),
        np.expand_dims(arrays['Y'], 1),
        np.expand_dims(arrays['Z'], 1)
    ], axis=-1).astype(np.float64)

valid_mask = ~np.isnan(dist)  # 找到有效点的索引

points_valid = points[valid_mask]
dist_valid = dist[valid_mask]

remove_mask = knn_outlier_removal(dist_valid)

dist_valid = dist_valid[remove_mask]
points_valid = points_valid[remove_mask]

# color = distance_to_color(dist_valid)

output_path = 'C:/Users/xhy/Desktop/pos1_pos3_difference_color.las'

# 创建las文件
# 1. Create a new header
header = laspy.LasHeader(point_format=3, version="1.2")
# header.offsets = np.min(cluster_points, axis=0)
# header.scales = np.array([1, 1, 1])

# 2. Create a Las
las = laspy.LasData(header)

las.add_extra_dim(laspy.ExtraBytesParams(
    name="distance",
    type=np.float32,  # 支持 np.float32 或 np.float64
    description="Distance from origin"
))

las.x = points_valid[:, 0]
las.y = points_valid[:, 1]
las.z = points_valid[:, 2]
# las.red = color[:, 0] * 255
# las.green = color[:, 1] * 255
# las.blue = color[:, 2] * 255
las.distance = dist_valid
las.write(output_path)
