import numpy as np
from pathlib import Path
from tqdm import tqdm
import pdal
import json
from sklearn.neighbors import KDTree
import pickle
import os

def normalize_attributes(array: np.array, min_value: float, max_value: float, mean_value: float, std_value: float):
    min_value = np.clip(mean_value - 2 * std_value, min_value, max_value)
    max_value = np.clip(mean_value + 2 * std_value, min_value, max_value)
    value = (array.astype(np.float32) - min_value) / (max_value - min_value)
    value = np.clip(value, 0.0, 1.0)
    return np.expand_dims(value, 1)


def process(cfg):
    cfg.data.root = os.path.abspath(cfg.data.root)
    processed_dir = os.path.join(cfg.data.root, f'processed-{cfg.data.grid_size:.2f}')
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)
    raw_root_dir = os.path.join(cfg.data.root)
    raw_root_dir = Path(raw_root_dir)
    paths = []
    for file_path in raw_root_dir.rglob('*.laz'):
        if file_path.is_file():
            paths.append(file_path)

    class_counts = None
    train = []
    test = []
    val = []
    # 1. 首先每个文件都生成一个 json 的索引，统计 intensity 最大最小值
    for raw_path in tqdm(paths, desc='Processing...'):
        raw_path = str(raw_path)
        name = os.path.splitext(os.path.basename(raw_path))[0]
        filename = name + '.npy'

        pth_path = os.path.join(processed_dir, filename)
        kdt_path = os.path.join(processed_dir, name + '_KDTree.pkl')

        if 'Train' in raw_path:
            train.append(name)
        elif 'Test' in raw_path:
            test.append(name)
        elif 'Validation' in raw_path:
            val.append(name)
        raw_path = raw_path.replace(os.sep, '/')

        pipeline = pdal.Pipeline()
        pipeline |= pdal.Reader.las(filename=raw_path)
        pipeline |= pdal.Filter.stats(dimensions='Intensity')
        pipeline |= pdal.Filter.voxelcenternearestneighbor(cell=cfg.data.grid_size)
        # outlier剔除场景中过高或过低的异常点
        pipeline |= pdal.Filter.outlier(method='statistical',
                                        multiplier=3.0)
        pipeline |= pdal.Filter.range(limits='Classification[1:2]')
        pipeline |= pdal.Filter.assign(value=["Classification = 0 WHERE Classification == 1",
                                              "Classification = 1 WHERE Classification == 2"])

        count = pipeline.execute()

        arrays = pipeline.arrays[0]
        metadata = pipeline.metadata['arradata']
        stats = metadata['filters.stats']['statistic'][0]

        bounds = metadata['readers.las']
        minx = bounds['minx']
        miny = bounds['miny']
        minz = bounds['minz']

        pos = np.concatenate([
            np.expand_dims(arrays['X'] - minx, 1),
            np.expand_dims(arrays['Y'] - miny, 1),
            np.expand_dims(arrays['Z'] - minz, 1)
        ], axis=-1).astype(np.float32)
        label = arrays['Classification']

        point_npy = np.zeros(pos.shape[0], dtype=[
            ('x', np.float32),  # X坐标 (浮点数)
            ('y', np.float32),  # Y坐标 (浮点数)
            ('z', np.float32),  # Z坐标 (浮点数)
            ('intensity', np.float32),  # 强度值 (浮点数)
            ('r', np.uint8),  # 红色分量 (0-255)
            ('g', np.uint8),  # 绿色分量 (0-255)
            ('b', np.uint8),  # 蓝色分量 (0-255)
            ('class', np.uint8)  # 类别分量 (int)
        ])

        if cfg.data.HasRGB:
            red = arrays['Red']
            green = arrays['Green']
            blue = arrays['Blue']

            point_npy['r'] = red
            point_npy['g'] = green
            point_npy['b'] = blue

        point_npy['x'] = (arrays['X'] - minx).astype(np.float32)
        point_npy['y'] = (arrays['Y'] - miny).astype(np.float32)
        point_npy['z'] = (arrays['Z'] - minz).astype(np.float32)
        point_npy['class'] = label

        if cfg.data.HasIntensity:
            stats = metadata['filters.stats']['statistic'][0]
            intensity = normalize_attributes(arrays['Intensity'], 0.0, (float)(stats['maximum']),
                                             stats['average'],
                                             stats['stddev'])
            point_npy['intensity'] = intensity.squeeze()


        np.save(pth_path, point_npy)

        kdtree = KDTree(pos)
        with open(kdt_path, 'wb') as f:
            pickle.dump(kdtree, f)

        _, counts = np.unique(label, return_counts=True)
        class_counts = class_counts + counts if class_counts is not None else counts

    splits = {
        'train': train,
        'val': val,
        'test': test,
        'class_weights': class_counts.tolist(),
    }
    json_path = os.path.join(processed_dir, 'metadata.json')
    with open(json_path, 'w') as f:
        json.dump(splits, f, indent=4)
