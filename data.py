# data_preprocessing.py
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Subset
import os

from utils import augment_rssi_with_noise
from models.base_models import train_gan_rssi, train_dae, apply_dae
from data_loader import WiFiDataset


def load_and_preprocess_data(train_data_path, val_data_path, seq_length=5,
                             train_subset_ratio=1.0, device='cuda'):
    # 路径检查
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"训练数据文件不存在：{train_data_path}")
    if not os.path.exists(val_data_path):
        raise FileNotFoundError(f"验证数据文件不存在：{val_data_path}")

    # 加载数据
    train_data = pd.read_csv(train_data_path)
    val_data = pd.read_csv(val_data_path)

    # 增强 Floor 4 样本
    print("🔁 开始增强 Floor 4 样本")
    train_data = augment_floor4_samples(train_data, device)

    # 标签修正
    train_data, val_data = adjust_labels(train_data, val_data)

    # 预处理坐标
    coord_train, coord_val, coord_mean, coord_std = preprocess_coordinates(train_data, val_data)

    # 训练和应用DAE
    print("Training DAE model...")
    dae = train_dae(
        features=train_data.filter(regex='^WAP').values,
        device=device
    )

    print("Applying DAE to train/val data...")
    denoised_train_features = apply_dae(
        dae,
        features=train_data.filter(regex='^WAP').values,
        device=device
    )

    denoised_val_features = apply_dae(
        dae,
        features=val_data.filter(regex='^WAP').values,
        device=device
    )

    # 创建数据集
    train_dataset = WiFiDataset(
        features=denoised_train_features,
        labels=train_data['FLOOR'].values,
        building_ids=train_data['BUILDINGID'].values,
        coordinates=coord_train,
        seq_length=seq_length,
        is_train=True
    )

    val_dataset = WiFiDataset(
        features=denoised_val_features,
        labels=val_data['FLOOR'].values,
        building_ids=val_data['BUILDINGID'].values,
        coordinates=coord_val,
        seq_length=seq_length,
        imputer=train_dataset.imputer,
        scaler=train_dataset.scaler,
        is_train=False
    )

    if train_subset_ratio < 1.0:
        subset_size = int(train_subset_ratio * len(train_dataset))
        train_dataset = Subset(train_dataset, range(subset_size))

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 计算floor_counts
    if isinstance(train_loader.dataset, Subset):
        floor_counts = np.bincount(train_loader.dataset.dataset.labels)
    else:
        floor_counts = np.bincount(train_loader.dataset.labels)

    return train_loader, val_loader, num_buildings, num_floors, floor_counts, coord_mean, coord_std


def augment_floor4_samples(train_data, device):
    """增强Floor 4样本"""
    floor4_data = train_data[train_data['FLOOR'] == 4].copy()
    if len(floor4_data) == 0:
        return train_data

    floor4_rssi = floor4_data.filter(regex='^WAP').values
    floor4_rssi[floor4_rssi == 100] = -104
    floor4_rssi = (floor4_rssi + 104) / 104 * 2 - 1

    # 训练GAN
    rssi_gan = train_gan_rssi(floor4_rssi, num_epochs=100, device=device)

    # 生成合成样本
    num_to_generate = 2000
    z = torch.randn(num_to_generate, 100).to(device)
    synthetic_rssi = rssi_gan(z).detach().cpu().numpy()
    synthetic_rssi = (synthetic_rssi + 1) / 2 * 104 - 104
    synthetic_rssi = np.clip(synthetic_rssi, -104, 0)

    synthetic_df = pd.DataFrame(synthetic_rssi, columns=train_data.filter(regex='^WAP').columns)
    synthetic_df['FLOOR'] = 4
    synthetic_df['BUILDINGID'] = 2

    # 使用KMeans分配坐标
    synthetic_df = assign_coordinates_to_synthetic(synthetic_df, floor4_data)

    # 合并到训练集
    train_data = pd.concat([train_data, synthetic_df], ignore_index=True)
    print(f"✅ Floor 4 样本增强后数量: {len(train_data[train_data['FLOOR'] == 4])}")

    return train_data


def assign_coordinates_to_synthetic(synthetic_df, floor4_data):
    """为合成样本分配坐标"""
    num_clusters = 4
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
    coords = floor4_data[['LONGITUDE', 'LATITUDE']].values
    cluster_labels = kmeans.fit_predict(coords)

    samples_per_cluster = len(synthetic_df) // num_clusters
    synthetic_parts = []

    for i in range(num_clusters):
        cluster_mask = cluster_labels == i
        cluster_coords = coords[cluster_mask]
        if len(cluster_coords) < 2:
            continue

        lon_mean, lon_std = cluster_coords[:, 0].mean(), cluster_coords[:, 0].std()
        lat_mean, lat_std = cluster_coords[:, 1].mean(), cluster_coords[:, 1].std()

        synth_part = synthetic_df.iloc[i * samples_per_cluster:(i + 1) * samples_per_cluster].copy()
        synth_part['LONGITUDE'] = np.random.normal(lon_mean, lon_std, size=len(synth_part))
        synth_part['LATITUDE'] = np.random.normal(lat_mean, lat_std, size=len(synth_part))
        synthetic_parts.append(synth_part)

    return pd.concat(synthetic_parts, ignore_index=True)


def adjust_labels(train_data, val_data):
    """调整标签从0开始"""
    for col in ['BUILDINGID', 'FLOOR']:
        train_min = train_data[col].min()
        train_data[col] -= train_min
        val_data[col] -= train_min

    # 验证
    for data, name in [(train_data, '训练集'), (val_data, '验证集')]:
        print(f"\n{name}标签范围:")
        print(f"Building ID范围: {data['BUILDINGID'].min()} - {data['BUILDINGID'].max()}")
        print(f"Floor ID范围: {data['FLOOR'].min()} - {data['FLOOR'].max()}")

    num_buildings = int(train_data['BUILDINGID'].max()) + 1
    num_floors = int(train_data['FLOOR'].max()) + 1

    print(f"\n模型配置:")
    print(f"建筑物数量: {num_buildings}")
    print(f"楼层数量: {num_floors}")

    return train_data, val_data


def preprocess_coordinates(train_data, val_data):
    """预处理坐标数据"""
    coord_train = train_data[['LONGITUDE', 'LATITUDE']].values
    coord_val = val_data[['LONGITUDE', 'LATITUDE']].values

    coord_mean = coord_train.mean(axis=0)
    coord_std = coord_train.std(axis=0)

    coord_train = (coord_train - coord_mean) / coord_std
    coord_val = (coord_val - coord_mean) / coord_std

    return coord_train, coord_val, coord_mean, coord_std