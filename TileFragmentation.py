import laspy
import numpy as np
import json
import os
import math
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from laspy import LazBackend

# 设置全局LAZ后端（解决多进程环境初始化问题）
os.environ["LASPY_LAZBACKEND"] = "lazrs"  # 或 "laszip"


def process_tile_chunk(args):
    """并行处理单个瓦片并启用LAZ压缩"""
    points_in_tile, header, tile_path, dimension_names = args
    try:
        # 创建新头文件（继承原始格式）
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
            setattr(tile_las, dim, points_in_tile[dim])

        tile_las.write(
            tile_path,
            do_compress=True,
            laz_backend=LazBackend.Lazrs # 或 LazBackend.LASZIP
        )
        return True
    except Exception as e:
        return f"Error processing {tile_path}: {str(e)}"


def tile_las_point_cloud_optimized(input_path, tile_size=100.0, chunk_size=10_000_000, output_dir="tiles", max_workers=None):
    """
    1. 分块流式读取避免内存溢出
    2. 空间索引加速瓦片归属计算
    3. 多进程并行处理瓦片
    4. LAZ压缩减少I/O和存储
    5. 增量元数据更新防中断
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    metadata_path = os.path.join(output_dir, f"metadata.json")

    # 初始化元数据（立即写入防中断）
    metadata = {"name": base_name}
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    with laspy.open(input_path) as reader:
        header = reader.header
        min_bound = np.array(header.mins)
        max_bound = np.array(header.maxs)

        # 计算网格划分
        x_range = max_bound[0] - min_bound[0]
        y_range = max_bound[1] - min_bound[1]
        num_cols = math.ceil(x_range / tile_size)
        num_rows = math.ceil(y_range / tile_size)

        # 更新元数据
        metadata.update({
            "tile_size": tile_size,
            "min_x": float(min_bound[0]),
            "min_y": float(min_bound[1]),
            "max_x": float(max_bound[0]),
            "max_y": float(max_bound[1]),
            "num_rows": num_rows,
            "num_cols": num_cols,
            "total_tiles": num_rows * num_cols,
            "actual_tiles": 0
        })

        # 预计算瓦片ID映射函数
        def get_tile_id(x, y):
            col = np.floor((x - min_bound[0]) / tile_size).astype(int)
            row = np.floor((y - min_bound[1]) / tile_size).astype(int)
            return row * num_cols + col

        # 分块处理点云（每块100万点）
        total_points = header.point_count
        chunks = reader.chunk_iterator(chunk_size)
        tile_buffers = {}
        dimension_names = [dim.name for dim in header.point_format.dimensions]

        # 第一遍：分块读取并分配点到瓦片
        for chunk in tqdm(chunks, total=math.ceil(total_points / chunk_size), desc="分块处理点云"):
            # 计算当前块中每个点所属的瓦片ID
            tile_ids = get_tile_id(chunk.x, chunk.y)

            # 将点分配到对应瓦片的缓冲区
            for tile_id in np.unique(tile_ids):
                mask = (tile_ids == tile_id)
                points_in_tile = {dim: getattr(chunk, dim)[mask] for dim in dimension_names}

                if tile_id not in tile_buffers:
                    tile_buffers[tile_id] = []
                tile_buffers[tile_id].append(points_in_tile)

        # 第二遍：并行处理瓦片
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for tile_id, chunks in tile_buffers.items():
                # 合并该瓦片的所有分块
                merged_points = {dim: np.concatenate([chunk[dim] for chunk in chunks])
                                 for dim in dimension_names}

                # 计算瓦片行列号
                tile_name = f"{base_name}-{tile_id}"
                laz_path = os.path.join(output_dir, f"{tile_name}.laz")

                # 提交并行任务
                args = (merged_points, header, laz_path, dimension_names)
                futures.append(executor.submit(process_tile_chunk, args))

            # 监控进度并处理结果
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result is not True:
                    print(f"⚠️ 处理失败: {result}")
                    # 这里可添加重试逻辑
                else:
                    metadata["actual_tiles"] += 1

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    return metadata


# 使用示例
if __name__ == "__main__":
    # 创建最小化头文件
    import os
    import laspy
    header = laspy.LasHeader(point_format=0, version="1.2")
    header.offsets = np.array([0, 0, 0])
    header.scales = np.array([0.01, 0.01, 0.01])

    # 写入空文件以初始化LAZ后端
    with laspy.open("dummy.laz", mode="w", header=header, laz_backend=laspy.LazBackend.Lazrs) as writer:
        writer.write_points([])  # 写入空点云
    input_cloud = "C:/Users/12641/Desktop/CDY Data/adjust_new_20250819-032432_00155_Lidar30_0001_1.las"  # 1亿+点云文件
    tile_size = 50.0  # 瓦片大小(米)
    output_directory = "tiled_output"
    # 根据硬件调整参数
    metadata = tile_las_point_cloud_optimized(
        input_cloud,
        tile_size=tile_size,
        output_dir=output_directory,
        chunk_size=10_000_000,
        max_workers=8  # 推荐设置为CPU核心数的75%
    )

    file_path = "dummy.laz"
    try:
        if os.path.exists(file_path):
            os.remove(file_path)  # 或 os.unlink(file_path)
        else:
            print(f"文件不存在: {file_path}")
    except Exception as e:
        print(f"删除失败: {e}")

    print(f"分割完成! 生成有效瓦片: {metadata['actual_tiles']}/{metadata['total_tiles']}")