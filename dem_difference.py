import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np


def align_dem(dem1_path, dem2_path, aligned_dem1_path, aligned_dem2_path):
    # 读取第一个 DEM
    with rasterio.open(dem1_path) as src1:
        src1_profile = src1.profile
        src1_data = src1.read(1).astype(np.float32)
        src1_crs = src1.crs or rasterio.crs.CRS.from_epsg(4326)  # 默认 WGS84
        src1_transform = src1.transform

    # 读取第二个 DEM
    with rasterio.open(dem2_path) as src2:
        src2_data = src2.read(1).astype(np.float32)
        src2_crs = src2.crs or src1_crs  # 强制使用与 dem1 相同的 CRS
        src2_transform = src2.transform

    # 强制对齐到 dem1 的分辨率和范围
    target_crs = src1_crs
    target_transform = src1_transform
    target_width = src1_profile['width']
    target_height = src1_profile['height']

    # 重新投影 dem2 到 dem1 的参数
    aligned_dem2 = np.zeros((target_height, target_width), dtype=np.float32)
    reproject(
        source=src2_data,
        destination=aligned_dem2,
        src_transform=src2_transform,
        src_crs=src2_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear
    )

    # 保存对齐后的 dem2
    kwargs = src1_profile.copy()
    kwargs.update({
        'crs': target_crs,
        'transform': target_transform,
        'width': target_width,
        'height': target_height,
        'nodata': -9999
    })
    with rasterio.open(aligned_dem2_path, 'w', **kwargs) as dst:
        dst.write(aligned_dem2, 1)

    # 保存对齐后的 dem1（可选）
    with rasterio.open(aligned_dem1_path, 'w', **src1_profile) as dst:
        dst.write(src1_data, 1)

    return aligned_dem1_path, aligned_dem2_path

def subtract_dem(aligned_dem1_path, aligned_dem2_path, output_path):
    # 读取第一个对齐后的 DEM
    with rasterio.open(aligned_dem1_path) as src1:
        dem1 = src1.read(1).astype(np.float32)
        profile = src1.profile  # ✅ 在 with 块内保存 profile
        nodata1 = src1.nodata
        if nodata1 is not None:
            dem1[dem1 == nodata1] = np.nan

    # 读取第二个对齐后的 DEM
    with rasterio.open(aligned_dem2_path) as src2:
        dem2 = src2.read(1).astype(np.float32)
        nodata2 = src2.nodata
        if nodata2 is not None:
            dem2[dem2 == nodata2] = np.nan

    # 确保形状一致
    assert dem1.shape == dem2.shape, "对齐失败，数组形状不一致！"

    # 计算差值
    diff = dem1 - dem2
    diff[np.isnan(diff)] = -9999  # 设置无效值

    # 更新 profile
    profile.update(dtype=rasterio.float32, nodata=-9999)

    # 保存差值结果
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(diff, 1)

    print(f"差值结果已保存至: {output_path}")


if __name__ == "__main__":
    dem1_path = "0_005m/pos1_dem0_005m.tif"
    dem2_path = "0_005m/pos3_dem0_005m.tif"
    output_path = "0_005m/dem_difference0_005m.tif"
    aligned_dem1, aligned_dem2 = align_dem(dem1_path, dem2_path, '0_005m/temp1.tif', '0_005m/temp2.tif')
    subtract_dem(aligned_dem1, aligned_dem2, output_path)