import laspy
import numpy as np
from scipy.spatial import Delaunay
import os
import time
import argparse
import logging
import multiprocessing as mp


def setup_logger():
    """设置日志记录器"""
    logger = logging.getLogger("LAS_to_Mesh")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件处理器
    fh = logging.FileHandler('las_to_mesh.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def read_las(las_path):
    """读取 LAS 文件并提取 XYZ 坐标"""
    start_time = time.time()

    try:
        las = laspy.read(las_path)
        logger.info(f"成功读取 LAS 文件: {os.path.basename(las_path)}")
        logger.info(f"文件版本: {las.header.version}, 点数量: {len(las.points)}")

        # 提取 XYZ 坐标
        xyz = np.vstack((las.x, las.y, las.z)).transpose()

        return xyz

    except Exception as e:
        logger.error(f"读取LAS文件失败: {str(e)}")
        raise


def downsample_points(xyz, target_density):
    """基于目标密度下采样点云"""
    if target_density <= 0:
        return xyz

    start_time = time.time()
    current_count = xyz.shape[0]

    # 计算点云边界和总面积
    min_pt = np.min(xyz[:, :2], axis=0)
    max_pt = np.max(xyz[:, :2], axis=0)
    area = (max_pt[0] - min_pt[0]) * (max_pt[1] - min_pt[1])

    # 计算目标点的数量
    target_count = int(area * target_density)

    if target_count >= current_count:
        logger.info("当前点密度低于目标密度，跳过下采样")
        return xyz

    # 随机选择目标数量的点
    indices = np.random.choice(current_count, target_count, replace=False)

    return xyz[indices]


def constrained_delaunay_triangulation(xyz, max_edge_length):
    """带最大边缘长度约束的 Delaunay 2.5D 三角剖分"""
    start_time = time.time()

    # 进行初始三角剖分
    tri = Delaunay(xyz[:, :2])
    triangles = tri.simplices

    # 如果未设置最大边缘长度约束，直接返回所有三角形
    if max_edge_length <= 0:
        return xyz, triangles

    original_triangle_count = len(triangles)

    # 过滤掉任何一条边超过最大长度的三角形
    valid_triangles = []

    for tri_idx in range(len(triangles)):
        triangle = triangles[tri_idx]
        edges_valid = True

        # 检查三角形的三边
        for i in range(3):
            p1_idx, p2_idx = triangle[i], triangle[(i + 1) % 3]
            edge_vector = xyz[p1_idx, :2] - xyz[p2_idx, :2]
            edge_length = np.linalg.norm(edge_vector)

            if edge_length > max_edge_length:
                edges_valid = False
                break

        if edges_valid:
            valid_triangles.append(triangle)

    return xyz, np.array(valid_triangles, dtype=np.int32)


def save_mesh(points, triangles, output_path):
    """将网格保存为 OBJ 或 FBX 文件"""
    start_time = time.time()

    try:
        # 根据文件扩展名确定输出格式
        file_ext = os.path.splitext(output_path)[1].lower()

        # 顶点数检查
        if len(points) > 5000000:
            logger.warning(f"顶点数巨大 ({len(points)}), 文件可能非常大")

        # OBJ 格式输出
        if file_ext == '.obj':
            with open(output_path, 'w') as f:
                # 写入顶点
                f.write(f"# Vertices: {len(points)}\n")
                for p in points:
                    f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

                # 写入面片
                f.write(f"# Faces: {len(triangles)}\n")
                for t in triangles:
                    # OBJ索引从1开始
                    f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")

            logger.info(f"OBJ文件保存成功: {output_path}")

        # FBX 格式输出（需要安装fbx-sdk）
        elif file_ext == '.fbx':
            try:
                import fbx

                # 初始化FBX SDK对象
                manager = fbx.FbxManager.Create()
                scene = fbx.FbxScene.Create(manager, "Scene")

                # 创建网格节点
                mesh_name = os.path.splitext(os.path.basename(output_path))[0]
                mesh_node = fbx.FbxNode.Create(manager, mesh_name)
                scene.GetRootNode().AddChild(mesh_node)

                # 创建网格对象
                fbx_mesh = fbx.FbxMesh.Create(manager, mesh_name)

                # 设置顶点
                fbx_mesh.InitControlPoints(len(points))
                control_points = fbx_mesh.GetControlPoints()
                for i, p in enumerate(points):
                    control_points[i] = fbx.FbxVector4(p[0], p[1], p[2])

                # 设置三角形
                for t in triangles:
                    fbx_mesh.BeginPolygon()
                    fbx_mesh.AddPolygon(t[0])
                    fbx_mesh.AddPolygon(t[1])
                    fbx_mesh.AddPolygon(t[2])
                    fbx_mesh.EndPolygon()

                # 设置法线（粗糙估计，实际应计算）
                normal_layer = fbx_mesh.CreateElementNormal()
                normal_layer.SetMappingMode(fbx.FbxLayerElement.eByControlPoint)
                normal_layer.SetReferenceMode(fbx.FbxLayerElement.eDirect)

                # 创建法线数组
                direct_array = normal_layer.GetDirectArray()
                for _ in points:
                    # 创建法线向量（实际应用应计算真实法线）
                    normal = fbx.FbxVector4(0, 0, 1)
                    direct_array.Add(normal)

                # 连接网格到节点
                mesh_node.SetNodeAttribute(fbx_mesh)
                mesh_node.SetShadingMode(fbx.FbxNode.eTextureShading)

                # 创建导出器
                exporter = fbx.FbxExporter.Create(manager, "")
                exporter.Initialize(output_path, -1, manager.GetIOSettings())
                exporter.Export(scene)

                logger.info(f"FBX文件保存成功: {output_path}")

            except ImportError:
                logger.error("FBX导出需要fbx-sdk库，请安装Autodesk FBX SDK")
                logger.warning("回退保存为OBJ格式...")
                new_path = os.path.splitext(output_path)[0] + '.obj'
                save_mesh(points, triangles, new_path)
                return

        # 不支持格式
        else:
            logger.error(f"不支持的格式: {file_ext}，使用.obj或.fbx扩展名")
            return False

        return True

    except Exception as e:
        logger.error(f"保存网格失败: {str(e)}")
        return False


def process_las_to_mesh(las_path, output_path, max_edge_length=2.0, target_density=1.0):
    """主处理函数：将 LAS 转换为网格模型"""
    global logger
    logger = setup_logger()

    total_start = time.time()

    logger.info("=" * 50)
    logger.info(f"输出格式: {os.path.splitext(output_path)[1]}")
    logger.info("=" * 50)

    try:
        # 步骤1: 读取LAS文件
        xyz = read_las(las_path)

        # 步骤2: 下采样点云（可选）
        if target_density > 0:
            xyz = downsample_points(xyz, target_density)

        # 步骤3: 执行带约束的Delaunay三角剖分
        xyz, triangles = constrained_delaunay_triangulation(xyz, max_edge_length)

        # 步骤4: 保存网格
        success = save_mesh(xyz, triangles, output_path)

        logger.info(f"处理完成! 总耗时: {time.time() - total_start:.2f}秒")
        return success

    except Exception as e:
        logger.error(f"处理过程中发生严重错误: {str(e)}")
        return False


if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='将LAS点云转换为Delaunay 2.5D网格模型')
    parser.add_argument("--input", "-i", default="input1_0.las", help="输入 LAS 文件路径")
    parser.add_argument("--output", "-o", default="terrain.obj", help="输出模型")
    parser.add_argument('--max_edge', type=float, default=30.0, help='最大三角形边缘长度(米)')
    parser.add_argument('--density', type=float, default=1.0, help='目标点密度(点/平方米)')

    args = parser.parse_args()

    # 执行处理
    success = process_las_to_mesh(
        las_path=args.input,
        output_path=args.output,
        max_edge_length=args.max_edge,
        target_density=args.density
    )

    if success:
        print(f"成功创建Delaunay 2.5D网格模型: {args.output}")
    else:
        print("处理失败，请查看日志文件 las_to_mesh.log 了解详情")