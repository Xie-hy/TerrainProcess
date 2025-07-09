import laspy
import numpy as np
from scipy.interpolate import griddata
from osgeo import gdal, osr
import argparse


def get_crs_from_las_file(las_file):
    """从 LAS 文件中提取坐标系 (CRS) 信息"""
    try:
        las = laspy.read(las_file)

        # 遍历所有 VLR 查找 CRS
        for vlr in las.vlrs:
            if hasattr(vlr, "record_id") and vlr.record_id == 2112:
                return vlr.crs_wkt

        # 如果没有找到 CRS VLR，尝试从文件头中获取
        if hasattr(las.header, "coordinate_reference_system"):
            return las.header.coordinate_reference_system

        return None  # 未找到有效坐标系信息

    except Exception as e:
        print(f"获取坐标系失败: {e}")
        return None


def create_dem(las_file, output_tif, resolution=1.0, method="nearest"):
    """从 LAS 点云生成 DEM"""
    # 读取 LAS 文件
    las = laspy.read(las_file)

    # 提取地面点（分类为 2）
    ground_mask = las.classification == 2
    if not np.any(ground_mask):
        raise ValueError("No ground points found in the LAS file.")

    x = las.x[ground_mask]
    y = las.y[ground_mask]
    z = las.z[ground_mask]

    # 确定范围
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # 计算网格尺寸
    width = int(np.ceil((x_max - x_min) / resolution))
    height = int(np.ceil((y_max - y_min) / resolution))

    # 生成网格坐标
    xi = np.linspace(x_min, x_min + width * resolution, width)
    yi = np.linspace(y_min, y_min + height * resolution, height)
    xi, yi = np.meshgrid(xi, yi)

    # 插值
    zi = griddata(points=np.column_stack((x, y)), values=z, xi=(xi, yi), method=method)

    # 获取坐标系
    crs_wkt = get_crs_from_las_file(las_file)

    # 设置默认坐标系（如果未找到）
    if crs_wkt is None:
        print("Warning: No CRS found in LAS file. Using WGS84 as default (may be incorrect).")
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)  # WGS84
    else:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(crs_wkt)

    # 创建 GeoTIFF
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_tif, xi.shape[1], yi.shape[0], 1, gdal.GDT_Float32)

    # 设置地理变换
    geo_transform = (x_min, resolution, 0, y_max, 0, -resolution)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(srs.ExportToWkt())

    # 写入数据
    dataset.GetRasterBand(1).WriteArray(zi)
    dataset.GetRasterBand(1).SetNoDataValue(-9999)  # 设置无效值
    dataset.FlushCache()
    dataset = None  # 关闭数据集


def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="从 LAS 点云生成 DEM")

    # 必填参数（无默认值）
    parser.add_argument("--input", "-i", default="input.las", help="输入 LAS 文件路径")
    parser.add_argument("--output", "-o", default="output_dem.tif", help="输出 GeoTIFF 文件路径（默认: output_dem.tif）")
    parser.add_argument("--resolution", "-r", type=float, default=1.0, help="DEM 分辨率（米，默认 1.0）")
    parser.add_argument("--method", "-m", choices=["nearest", "linear", "cubic"], default="nearest",
                        help="插值方法（默认: nearest）")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数生成 DEM
    try:
        create_dem(args.input, args.output, resolution=args.resolution, method=args.method)
        print(f"DEM 生成成功，输出文件: {args.output}")
    except Exception as e:
        print(f"生成 DEM 失败: {e}")


if __name__ == "__main__":
    main()