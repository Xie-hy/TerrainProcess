import csv
import json
import math
import os.path
import numpy as np
import laspy
from tqdm import tqdm
from laspy import LazBackend

# 设置全局LAZ后端（解决多进程环境初始化问题）
os.environ["LASPY_LAZBACKEND"] = "lazrs"  # 或 "laszip"

def search_file(base_path):
    """
    检查与base_path同名但扩展名为.laz或.las的文件是否存在，

    参数:
        base_path: 文件的基础路径（不包含扩展名）
    """
    # 构造完整的文件路径
    laz_file = base_path + '.laz'
    las_file = base_path + '.las'

    if os.path.exists(laz_file):
        return laz_file
    elif os.path.exists(las_file):
        return las_file
    else:
        print(f"未找到{base_path}对应的.laz或.las文件")
        return None


if __name__ == "__main__":
    # 初始化变量，初始值设为None或第一行的值
    vertex_x_min = None
    vertex_x_max = None
    vertex_y_min = None
    vertex_y_max = None

    with open('boundary.csv', 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)  # 使用DictReader方便通过列名访问

        for row in reader:
            # 假设x字段和y字段是数字，转换为浮点数进行比较
            try:
                x_val = float(row['x'])
                y_val = float(row['y'])
            except ValueError:
                # 处理无法转换的数据（例如空值或非数字字符串）
                print(f"跳过无效行: {row}")
                continue

            # 查找x字段的最大最小值
            if vertex_x_min is None or x_val < vertex_x_min:
                vertex_x_min = x_val
            if vertex_x_max is None or x_val > vertex_x_max:
                vertex_x_max = x_val

            # 查找y字段的最大最小值
            if vertex_y_min is None or y_val < vertex_y_min:
                vertex_y_min = y_val
            if vertex_y_max is None or y_val > vertex_y_max:
                vertex_y_max = y_val

    # 打印结果
    print(f"x 最小值: {vertex_x_min}, 最大值: {vertex_x_max}")
    print(f"y 最小值: {vertex_y_min}, 最大值: {vertex_y_max}")

    tile_file_dir = 'tiled_output'
    tile_metadata_file_path = os.path.join(tile_file_dir, 'metadata.json')
    output_path = 'search_result.laz'

    # 使用 with open 语句安全地打开文件
    with open(tile_metadata_file_path, 'r', encoding='utf-8') as file:  # 指定正确的编码
        data = json.load(file)

    pc_name = data['name']
    tile_min_x = data['min_x']
    tile_max_x = data['max_x']
    tile_min_y = data['min_y']
    tile_max_y = data['max_y']

    tile_size = data['tile_size']
    num_cols = data['num_cols']

    min_col = math.floor((vertex_x_min - tile_min_x) / tile_size)
    min_row = math.floor((vertex_y_min - tile_min_y) / tile_size)
    max_col = math.floor((vertex_x_max - tile_min_x) / tile_size)
    max_row = math.floor((vertex_y_max - tile_min_y) / tile_size)

    tiles_ids = []
    for row in range(min_row, max_row + 1):
        for col in range(min_col, max_col + 1):
            tile_idx = row * num_cols + col
            tiles_ids.append(tile_idx)

    all_points = []
    for tile_id in tqdm(tiles_ids, desc="Processing Tiles", unit="tile"):
        temp_path = os.path.join(tile_file_dir, f'{pc_name}-{tile_id}')
        pc_path = search_file(temp_path)
        if pc_path is None:
            print("point cloud path is not exist!")
            continue
        with laspy.open(pc_path) as reader:
            # 读取第一个文件并创建输出文件框架
            las_data = reader.read()
            header = las_data.header
            # 创建输出文件
            out_file = laspy.LasData(header)
            dimension_names = [dim.name for dim in header.point_format.dimensions]
            points_in_tile = {dim: getattr(las_data, dim) for dim in dimension_names}
            # 收集第一个文件的点数据
            all_points.append(points_in_tile)

    # 合并所有点数据
    print('Merging...')
    merged_points = {dim: np.concatenate([tile[dim] for tile in all_points])
                        for dim in dimension_names}

    print(f'Writing to {output_path} ...')
    # 将合并后的点写入输出文件
    new_header = laspy.LasHeader(
        point_format=header.point_format,
        version=header.version
    )
    new_header.scales = header.scales
    new_header.offsets = header.offsets

    # 创建点云对象
    tile_las = laspy.LasData(new_header)

    # 复制所有维度数据
    for dim in dimension_names:
        setattr(tile_las, dim, merged_points[dim])

    tile_las.write(
        output_path,
        do_compress=True,
        laz_backend=LazBackend.Lazrs  # 或 LazBackend.LASZIP
    )