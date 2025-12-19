import math
# from sched import scheduler
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR
# from torch.optim.lr_scheduler import OneCycleLR
from sched import scheduler as SchedScheduler
import seaborn as sns
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import KNNImputer
from torch.utils.data import Dataset, DataLoader, Subset, dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import os
import logging
from sklearn.metrics import classification_report, confusion_matrix
# from models import WiFiLocalizationModel, DAE
from tqdm import tqdm

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
# 改为强制使用GPU（如果不可用则报错）
assert torch.cuda.is_available(), "CUDA is not available - GPU required!"
device = torch.device('cuda')
print(f"Using device: {device}")

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 自定义初始化方法
def my_init_sigmoid(shape):
    rnd = torch.rand(shape)
    fan_in = shape[0]
    fan_out = shape[1] if len(shape) > 1 else 1
    return 8. * (rnd - 0.5) * np.sqrt(6) / np.sqrt(fan_in + fan_out)


def my_init_others(shape):
    rnd = torch.rand(shape)
    fan_in = shape[0]
    return 2. * (rnd - 0.5) / np.sqrt(fan_in)


import torch.nn as nn
import torch.nn.functional as F


# 简单生成器
class RSSIGenerator(nn.Module):
    def __init__(self, noise_dim=100, output_dim=520):
        super(RSSIGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()  # 输出范围约束
        )

    def forward(self, z):
        return self.model(z)


# 判别器
class RSSIDiscriminator(nn.Module):
    def __init__(self, input_dim=520):
        super(RSSIDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def train_gan_rssi(floor4_data, num_epochs=200, batch_size=64, noise_dim=100, device='cuda'):
    floor4_data = torch.tensor(floor4_data, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(floor4_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator = RSSIGenerator(noise_dim=noise_dim, output_dim=floor4_data.shape[1]).to(device)
    discriminator = RSSIDiscriminator(input_dim=floor4_data.shape[1]).to(device)

    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-3)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        for real_batch, in loader:
            batch_size = real_batch.size(0)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train discriminator
            z = torch.randn(batch_size, noise_dim).to(device)
            fake_data = generator(z)
            d_loss = criterion(discriminator(real_batch), real_labels) + \
                     criterion(discriminator(fake_data.detach()), fake_labels)

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # Train generator
            z = torch.randn(batch_size, noise_dim).to(device)
            fake_data = generator(z)
            g_loss = criterion(discriminator(fake_data), real_labels)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

    return generator


def extract_dae_features(dae, features, scaler):
    features = features.copy()
    features[features == 100] = -105
    features = scaler.transform(features)
    features_tensor = torch.FloatTensor(features).to(device)
    dae.eval()
    with torch.no_grad():
        encoded = dae.encoder(features_tensor)
    return encoded.cpu().numpy()


# 数据预处理类
class WiFiDataset(Dataset):
    def __init__(self, data_path=None, features=None, labels=None,
                 coordinates=None, building_ids=None, seq_length=5,
                 imputer=None, scaler=None, columns_to_drop=None,  # 新增参数
                 is_train=True):  # 新增训练/验证标志
        self.seq_length = seq_length
        self.is_train = is_train  # 控制是否数据增强

        # 数据加载（保持原有逻辑）
        if data_path is not None:
            data = pd.read_csv(data_path)
            self.features = data.filter(regex='^WAP\d+').values
            self.labels = data['FLOOR'].values
            self.coordinates = data[['LONGITUDE', 'LATITUDE']].values
            self.building_ids = data['BUILDINGID'].values
        elif features is not None:
            self.features = features
            self.labels = labels
            self.coordinates = coordinates
            self.building_ids = building_ids if building_ids is not None else np.zeros(len(labels))
        else:
            raise ValueError("需要提供 data_path 或 features/labels/coordinates 等")

        # 确保float类型
        self.features = self.features.astype(float)

        # === 关键修改1：缺失值处理 ===
        self.features[self.features == 100] = -105  # 论文推荐值而非np.nan

        # # 仅训练集计算需要删除的列
        # if is_train and columns_to_drop is None:
        #     missing_values = pd.DataFrame(self.features).isna().mean() * 100
        #     threshold = 95
        #     self.columns_to_drop = missing_values[missing_values > threshold].index
        # else:
        #     self.columns_to_drop = columns_to_drop  # 验证集使用训练集的columns_to_drop
        #
        # # 删除高缺失率列
        # if self.columns_to_drop is not None:
        #     self.features = np.delete(self.features, self.columns_to_drop, axis=1)

        # === 关键修改2：KNN填充 ===
        self.imputer = None  # 不进行填充
        # === 关键修改3：标准化 ===
        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)  # 训练集计算mean/std
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)  # 验证集使用训练集的scaler

        # === 关键修改4：数据增强（仅训练集）===
        # if is_train:
        #     if self.labels[floor] in [0, 1, 2]:  # 主类
        #         self.features = self.add_noise(self.features, 0.2)
        #         self.features = channel_dropout(self.features, dropout_rate=0.1)
        #     else:  # 少数类，不遮挡
        #         self.features = self.add_noise(self.features, 0.05)

        # 7. 时序参数
        self.seq_length = seq_length

        # 8. 数据验证
        assert len(self.features) == len(self.labels), "特征与标签数量不匹配"
        assert len(self.features) == len(self.coordinates), "特征与坐标数量不匹配"
        if self.building_ids is not None:
            assert len(self.features) == len(self.building_ids), "特征与建筑ID数量不匹配"

        # 计算num_classes
        self.num_buildings = len(np.unique(self.building_ids)) if self.building_ids is not None else 1
        self.num_floors = len(np.unique(self.labels))

        # 模型配置
        model_config = {
            'input_dim': 520,
            'spatial_feature_dim': 256,
            'temporal_hidden_dim': 128,
            'temporal_feature_dim': 256,
            'eca_channels': 512,
            'num_classes': None,
        }

        # 如果 building_ids 为 None，则设置为全零数组
        if self.building_ids is None:
            self.building_ids = np.zeros_like(self.labels)

        print(f"修正后标签范围: B={self.building_ids.min()}-{self.building_ids.max()}, "
              f"F={self.labels.min()}-{self.labels.max()}, ")

    def __len__(self):
        return len(self.features) - self.seq_length + 1

    def add_noise(self, data, noise_level=0.2):
        """添加高斯噪声进行数据增强"""
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise

    def __getitem__(self, idx):

        # 获取序列特征
        seq_features = self.features[idx:idx + self.seq_length]

        # 获取序列标签
        seq_labels = self.labels[idx:idx + self.seq_length]

        # 获取序列坐标
        seq_coordinates = self.coordinates[idx:idx + self.seq_length]

        # 获取序列建筑ID
        if self.building_ids is not None:
            seq_building_ids = self.building_ids[idx:idx + self.seq_length]
        else:
            seq_building_ids = np.zeros(self.seq_length)  # 如果没有建筑ID，使用0填充

        # 创建时间戳
        timestamps = np.arange(self.seq_length)

        # 计算位置变化
        position_changes = np.sqrt(
            np.sum(np.diff(seq_coordinates, axis=0) ** 2, axis=1)
        )
        position_changes = np.pad(position_changes, (1, 0))  # 在开始处填充0

        # 转换为张量
        features = torch.FloatTensor(seq_features)
        coordinates = torch.FloatTensor(seq_coordinates)
        hierarchical_labels = torch.LongTensor(np.column_stack((seq_building_ids, seq_labels)))
        # cluster_labels = torch.LongTensor(seq_space_ids)  # 使用空间ID作为聚类标签
        timestamps = torch.FloatTensor(timestamps)
        position_changes = torch.FloatTensor(position_changes)

        return features, coordinates, hierarchical_labels, timestamps, position_changes


print("开始构建数据集...")

print("数据集构建完成")


class DAE(nn.Module):
    def __init__(self, input_dim, encoding_dim=128):
        super(DAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# 修改DAE处理部分，保持一致的标准化
# 修改DAE处理部分，保持一致的标准化
def train_dae(features, encoding_dim=128, noise_std=0.02, epochs=30):
    features = features.copy()
    features[features == 100] = -105  # 保持与后续处理一致
    # 使用StandardScaler而不是简单的线性归一化
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    # ...其余代码不变...
    # 构建输入张量
    features_tensor = torch.FloatTensor(features).to(device)

    input_dim = features.shape[1]
    dae = DAE(input_dim=input_dim, encoding_dim=encoding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(dae.parameters(), lr=1e-3)

    # 训练
    dae.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        noise = torch.randn_like(features_tensor) * noise_std
        noisy_input = torch.clamp(features_tensor + noise, 0, 1)
        outputs = dae(noisy_input)
        loss = criterion(outputs, features_tensor)
        loss.backward()
        optimizer.step()
    return dae


def apply_dae(dae, features):
    """使用训练好的DAE去噪"""
    features = features.copy()
    features[features == 100] = -105
    features = (features + 105) / 105  # 归一化到0~1

    features_tensor = torch.FloatTensor(features).to(device)

    dae.eval()
    with torch.no_grad():
        outputs = dae(features_tensor)

    # 还原回原始RSSI范围
    denoised = outputs.cpu().numpy() * 105 - 105
    return denoised


# ECA模块 (基于论文中的实现)
class ECAModule(nn.Module):
    def __init__(self, channels=128, gamma=2, b=1):
        super(ECAModule, self).__init__()
        self.channels = channels
        self.gamma = gamma
        self.b = b

        # 自适应确定卷积核大小
        t = int(abs((np.log2(self.channels) + self.b) / self.gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

    def forward(self, x):
        # 输入x的形状: (batch_size, seq_length, channels)
        batch_size, seq_length, channels = x.size()

        # 重塑为(batch_size * seq_length, channels)
        x_reshaped = x.reshape(-1, channels)

        # 添加通道维度 -> (batch_size * seq_length, 1, channels)
        x_channel = x_reshaped.unsqueeze(1)

        # 通道注意力
        y = self.conv(x_channel)  # shape: (batch_size * seq_length, 1, channels)
        y = torch.sigmoid(y).squeeze(1)  # shape: (batch_size * seq_length, channels)

        # 加权输入
        y = y.reshape(batch_size, seq_length, channels)

        # 返回原始输入乘以注意力权重
        return x * y


# 1D-CNN网络 (用于空间特征提取)
class SpatialFeatureExtractor(nn.Module):
    def __init__(self, input_dim=520, feature_dim=256):
        super(SpatialFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, feature_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(feature_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self._init_weights()  # 添加自定义初始化

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                shape = m.weight.shape
                m.weight.data = my_init_others(shape)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                shape = m.weight.shape
                m.weight.data = my_init_sigmoid(shape)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x形状: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()

        # 对每个时间步单独处理
        spatial_features = []
        for t in range(seq_len):
            x_t = x[:, t, :].unsqueeze(1)  # (batch_size, 1, input_dim)

            # 1D卷积处理
            out = self.relu(self.bn1(self.conv1(x_t)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.relu(self.bn3(self.conv3(out)))

            # 全局平均池化
            out = self.pool(out).squeeze(-1)  # (batch_size, feature_dim)
            spatial_features.append(out.unsqueeze(1))

        # 合并所有时间步
        spatial_features = torch.cat(spatial_features, dim=1)  # (batch_size, seq_len, feature_dim)
        return spatial_features


# GRU网络 (用于时间特征提取)
class TemporalFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, feature_dim):
        super(TemporalFeatureExtractor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, feature_dim)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                shape = m.weight.shape
                m.weight.data = my_init_sigmoid(shape)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x形状: (batch_size, seq_len, input_dim)
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, 2*hidden_dim)

        # 对每个时间步的特征进行变换
        temporal_features = self.relu(self.fc(gru_out))  # (batch_size, seq_len, feature_dim)
        return temporal_features


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 权重，可以传入 floor_weights
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()


# 主模型
class WiFiLocalizationModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 增加专门的楼层特征提取层
        self.floor_specific_extractor = nn.Sequential(
            nn.Linear(config['temporal_feature_dim'], 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )

        # 修改分类头
        self.floor_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config['num_floors'])
        )

        self.attn_weights = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.config = config
        self.building_classifier = nn.Linear(512, config['num_buildings'])
        self.floor_classifier = nn.Linear(512, config['num_floors'])
        # self.coord_regressor = nn.Linear(512, 2)  # 回归经纬度（LONGITUDE, LATITUDE）
        # 增加更深的坐标回归器
        # 更强大的坐标回归器
        # 加强坐标回归器（在WiFiLocalizationModel中）
        self.coord_regressor = nn.Sequential(
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 2)
        )

        # 添加多尺度时空特征融合
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 增加更强大的特征提取层
        self.spatial_extractor = nn.Sequential(
            SpatialFeatureExtractor(input_dim=config['input_dim'], feature_dim=config['spatial_feature_dim']),
            nn.LayerNorm(config['spatial_feature_dim']),
            nn.Dropout(0.5)
        )

        # 增强时间特征提取
        self.temporal_extractor = nn.Sequential(
            TemporalFeatureExtractor(
                input_dim=config['input_dim'],
                hidden_dim=config['temporal_hidden_dim'],
                feature_dim=config['temporal_feature_dim']
            ),
            nn.LayerNorm(config['temporal_feature_dim']),
            nn.Dropout(0.5)
        )
        # ECA模块
        self.eca = nn.Sequential(
            ECAModule(channels=config['eca_channels']),
            nn.Dropout(0.3)
        )

        # 全连接层及归一化
        self.fc1 = nn.Linear(config['eca_channels'], 512)
        self.layer_norm = nn.LayerNorm(512)  # 🔁 替代 BatchNorm1d
        self.fc2 = nn.Linear(512, config['num_classes'])

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.7)

        # GMM参数
        self.num_classes = config['num_classes']
        self.gmm = None

    # 替换原来的 forward
    def forward(self, x, return_features=False):
        spatial_features = self.spatial_extractor(x)
        temporal_features = self.temporal_extractor(x)
        shared_features = torch.cat([spatial_features, temporal_features], dim=2)
        shared_features = self.eca(shared_features)

        # 下游分支
        fc_out = self.relu(self.layer_norm(self.fc1(shared_features)))
        fc_out = self.dropout1(fc_out)
        fc_out = self.dropout2(fc_out)  # ✅ 添加这一行

        # 分类：逐时间步
        building_logits = self.building_classifier(fc_out)
        floor_logits = self.floor_classifier(fc_out)

        # 回归：平均池化后输出坐标
        # 计算 attention 权重
        attn_scores = self.attn_weights(fc_out).squeeze(-1)  # (B, T)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, T, 1)

        # 加权求和替代 mean pooling
        attn_pooled = torch.sum(fc_out * attn_weights, dim=1)  # (B, 512)
        coord_pred = self.coord_regressor(attn_pooled)

        if return_features:
            return (building_logits, floor_logits, coord_pred), shared_features
        return building_logits, floor_logits, coord_pred


def train_epoch(model, train_loader, criterion, floor_loss_fn, optimizer, device, coord_mean, coord_std,
                flood_level=0.3):
    model.train()
    total_loss = 0
    coord_mae = 0

    if not isinstance(coord_mean, torch.Tensor):
        coord_mean = torch.tensor(coord_mean, dtype=torch.float32, device=device)
    if not isinstance(coord_std, torch.Tensor):
        coord_std = torch.tensor(coord_std, dtype=torch.float32, device=device)

    all_true_building, all_pred_building = [], []
    all_true_floor, all_pred_floor = [], []

    for features, coordinates, hierarchical_labels, _, _ in train_loader:
        features = features.to(device)
        coordinates = coordinates.to(device)
        hierarchical_labels = hierarchical_labels.to(device)

        building_outputs, floor_outputs, coord_outputs = model(features)

        # === 分类损失（建筑 & 楼层）===
        building_loss = criterion(
            building_outputs.reshape(-1, model.config['num_buildings']),
            hierarchical_labels[:, :, 0].reshape(-1)
        )
        floor_loss = floor_loss_fn(
            floor_outputs.reshape(-1, model.config['num_floors']),
            hierarchical_labels[:, :, 1].reshape(-1)
        )

        # === 坐标回归损失（在标准化空间中计算）===
        coord_target = coordinates.mean(dim=1)
        mae_loss = nn.L1Loss()(coord_outputs, coord_target)
        mse_loss = nn.MSELoss()(coord_outputs, coord_target)
        coord_loss = 0.7 * mae_loss + 0.3 * torch.sqrt(mse_loss + 1e-6)

        # === 坐标误差监控（在真实坐标空间中，单位：米）===
        coord_outputs_real = coord_outputs * coord_std + coord_mean
        coord_target_real = coord_target * coord_std + coord_mean
        errors = torch.norm(coord_outputs_real - coord_target_real, dim=1)
        coord_mae += torch.mean(errors).item()

        # === 总损失 ===
        loss = 0.3 * building_loss + 1.0 * floor_loss + 0.7 * coord_loss  # 权重可调
        # ====== 新增：Flooding 技术 ======
        if flood_level > 0:
            loss = (loss - flood_level).abs() + flood_level  # Flooding 公式
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # === 记录预测 ===
        _, building_pred = torch.max(building_outputs[:, -1, :], dim=1)
        _, floor_pred = torch.max(floor_outputs[:, -1, :], dim=1)
        all_true_building.extend(hierarchical_labels[:, -1, 0].cpu().numpy())
        all_pred_building.extend(building_pred.cpu().numpy())
        all_true_floor.extend(hierarchical_labels[:, -1, 1].cpu().numpy())
        all_pred_floor.extend(floor_pred.cpu().numpy())

    # === 精度统计 ===
    building_acc = accuracy_score(all_true_building, all_pred_building)
    floor_acc = accuracy_score(all_true_floor, all_pred_floor)

    return {
        'loss': total_loss / len(train_loader),
        'building_acc': building_acc,
        'floor_acc': floor_acc,
        'coord_mae': coord_mae / len(train_loader)
    }


def validate_epoch(model, val_loader, criterion, floor_loss_fn, device, coord_mean, coord_std):
    model.eval()
    total_loss = 0
    coord_mae = 0
    all_errors = []

    coord_mean = torch.tensor(coord_mean, dtype=torch.float32, device=device)
    coord_std = torch.tensor(coord_std, dtype=torch.float32, device=device)

    all_true_building, all_pred_building = [], []
    all_true_floor, all_pred_floor = [], []

    with torch.no_grad():
        for batch_idx, (features, coordinates, hierarchical_labels, _, _) in enumerate(val_loader):
            features = features.to(device)
            coordinates = coordinates.to(device)
            hierarchical_labels = hierarchical_labels.to(device)

            # 前向传播
            building_outputs, floor_outputs, coord_outputs = model(features)

            # --- 分类预测 ---
            _, building_pred = torch.max(building_outputs[:, -1, :], dim=1)
            _, floor_pred = torch.max(floor_outputs[:, -1, :], dim=1)

            # === 核心修改：对Floor 4的预测进行建筑约束修正 ===
            floor4_mask = (floor_pred == 4)  # 找到所有预测为Floor 4的样本
            if torch.any(floor4_mask):
                # 假设建筑2才有Floor 4（根据数据集实际情况调整）
                invalid_building = (building_pred[floor4_mask] != 2)
                # 将"建筑不是2但预测为Floor 4"的样本修正为建筑的最高楼层
                floor_pred[floor4_mask] = torch.where(
                    invalid_building,
                    3,  # 修正为楼层3（假设建筑0/1的最高楼层是3）
                    floor_pred[floor4_mask]  # 否则保持原预测
                )

            all_true_building.extend(hierarchical_labels[:, -1, 0].cpu().numpy())
            all_pred_building.extend(building_pred.cpu().numpy())
            all_true_floor.extend(hierarchical_labels[:, -1, 1].cpu().numpy())
            all_pred_floor.extend(floor_pred.cpu().numpy())

            # --- 坐标误差计算 ---
            # 替换为（序列末尾位置）：
            coord_target = coordinates.mean(dim=1)  # 与训练一致
            coord_outputs_real = coord_outputs * coord_std + coord_mean
            coord_target_real = coord_target * coord_std + coord_mean
            errors = torch.norm(coord_outputs_real - coord_target_real, dim=1)
            coord_mae += torch.mean(errors).item()
            all_errors.extend(errors.cpu().numpy())  # ✅ 新增

            # 损失计算（仅用于监控）
            building_loss = criterion(
                building_outputs.reshape(-1, model.config['num_buildings']),
                hierarchical_labels[:, :, 0].reshape(-1)
            )
            floor_loss = nn.CrossEntropyLoss()(
                floor_outputs.reshape(-1, model.config['num_floors']),
                hierarchical_labels[:, :, 1].reshape(-1)
            )
            coord_loss = nn.MSELoss()(coord_outputs, coord_target)
            total_loss += (0.3 * building_loss + 1.0 * floor_loss + 0.7 * coord_loss).item()

    # 分类报告
    print("\n📊 分类报告：")
    print(classification_report(all_true_floor, all_pred_floor,
                                target_names=["Floor 0", "Floor 1", "Floor 2", "Floor 3", "Floor 4"]))

    return {
        'loss': total_loss / len(val_loader),
        'building_acc': accuracy_score(all_true_building, all_pred_building),
        'floor_acc': accuracy_score(all_true_floor, all_pred_floor),
        'coord_mae': coord_mae / len(val_loader),  # 真实MAE（米）
        'errors': all_errors  # ✅ 新增
    }


def train_model(model, train_loader, val_loader, criterion, optimizer,
                floor_loss_fn, coord_mean, coord_std, scheduler,
                num_epochs, patience, device, flood_level=0.3):
    model = model.to(device)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    # 添加记录容器
    history = {
        'train_building': [],
        'val_building': [],
        'train_floor': [],
        'val_floor': []
    }

    # ✅ 初始化 EMA 参数字典
    ema_decay = 0.999
    shadow_params = {name: param.data.clone() for name, param in model.named_parameters()}

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        # 训练一个epoch
        train_metrics = train_epoch(model, train_loader, criterion, floor_loss_fn,
                                    optimizer, device, coord_mean, coord_std, flood_level=flood_level)
        train_losses.append(train_metrics['loss'])

        # 验证
        val_metrics = validate_epoch(model, val_loader, criterion, floor_loss_fn,
                                     device, coord_mean, coord_std)
        val_losses.append(val_metrics['loss'])
        scheduler.step()
        # 记录指标（放在打印指标之后）
        history['train_building'].append(train_metrics['building_acc'])
        history['val_building'].append(val_metrics['building_acc'])
        history['train_floor'].append(train_metrics['floor_acc'])
        history['val_floor'].append(val_metrics['floor_acc'])

        # 更新EMA参数
        for name, param in model.named_parameters():
            if name in shadow_params:
                shadow_params[name].mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

        # 调整学习率
        # scheduler.step(val_metrics['loss'])

        # ✅ 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.6f}")

        # 打印指标
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_metrics["loss"]:.4f}')
        print(f'Val Loss: {val_metrics["loss"]:.4f}')
        # 在train_model函数中修改打印部分
        print(f'Building Acc: {train_metrics["building_acc"]:.4f}/{val_metrics["building_acc"]:.4f}')
        print(f'Floor Acc: {train_metrics["floor_acc"]:.4f}/{val_metrics["floor_acc"]:.4f}')
        # print(f'Coord RMSE: {train_metrics["coord_rmse"]:.2f}/{val_metrics["coord_rmse"]:.2f}')
        # print(f'Coord MSE: {train_metrics["coord_mse"]:.2f}/{val_metrics["coord_mse"]:.2f}')
        print(f'Coord MAE (Mean Positioning Error): {train_metrics["coord_mae"]:.2f}/{val_metrics["coord_mae"]:.2f}')

        # ✅ 添加内存监控
        mem = psutil.virtual_memory()
        print(f"Memory usage: {mem.percent}%")

        # === 早停检查 + 保存最佳模型（基于EMA）===
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            # 保存当前正常参数模型作为备份
            torch.save(model.state_dict(), 'best_model_normal.pth')

            # 将 shadow EMA 参数赋值给模型再保存
            backup = {name: param.data.clone() for name, param in model.named_parameters()}
            for name, param in model.named_parameters():
                if name in shadow_params:
                    param.data.copy_(shadow_params[name])
            torch.save(model.state_dict(), 'best_model.pth')
            print("✅ EMA平滑参数已保存为 best_model.pth")

            # 恢复原参数，继续训练
            for name, param in model.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name])
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

    # 损失曲线绘图
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig('loss_curve.png')
    print("✅ 损失曲线已保存到 loss_curve.png")

    # 训练结束后绘制精度曲线（放在return之前）
    plt.figure(figsize=(10, 4))

    # 建筑分类精度曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_building'], 'b-', label='Train')
    plt.plot(history['val_building'], 'r--', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Building Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 楼层分类精度曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_floor'], 'b-', label='Train')
    plt.plot(history['val_floor'], 'r--', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Floor Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 训练曲线已保存为 training_curves.png")

    # ✅ 新增绘制 CDF 图
    val_errors = val_metrics.get('errors', None)
    if val_errors is not None:
        errors_np = np.sort(np.array(val_errors))
        cdf = np.arange(len(errors_np)) / len(errors_np)

        plt.figure(figsize=(8, 6))
        plt.plot(errors_np, cdf, label='CDF of Localization Error')
        plt.xlabel('Localization Error (meters)')
        plt.ylabel('Cumulative Probability')
        plt.title('CDF of Validation Localization Error')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('val_error_cdf.png')
        print("✅ 验证集定位误差CDF图已保存为 val_error_cdf.png")

    return train_losses, val_losses


#
# # 数据加载和预处理
def augment_rssi_with_noise(df, noise_std=2.0):
    wap_columns = df.filter(regex='^WAP').columns
    df_aug = df.copy()
    noise = np.random.normal(0, noise_std, size=df_aug[wap_columns].shape)
    df_aug[wap_columns] = df_aug[wap_columns] + noise
    df_aug[wap_columns] = df_aug[wap_columns].clip(-105, 0)
    return df_aug


def load_and_preprocess_data(train_data_path='/kaggle/input/UjiIndoorLoc/TrainingData.csv',
                             val_data_path='/kaggle/input/UjiIndoorLoc/ValidationData.csv',
                             seq_length=5, train_subset_ratio=1.0):
    # 路径检查
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"训练数据文件不存在：{train_data_path}")
    if not os.path.exists(val_data_path):
        raise FileNotFoundError(f"验证数据文件不存在：{val_data_path}")

    # 加载数据
    train_data = pd.read_csv(train_data_path)
    val_data = pd.read_csv(val_data_path)

    from sklearn.cluster import KMeans

    print("🔁 开始增强 Floor 4 样本")

    floor4_data = train_data[train_data['FLOOR'] == 4].copy()
    floor4_rssi = floor4_data.filter(regex='^WAP').values

    floor4_rssi[floor4_rssi == 100] = -105
    floor4_rssi = (floor4_rssi + 105) / 105 * 2 - 1  # [-1, 1]

    # ✅ 训练 GAN
    rssi_gan = train_gan_rssi(floor4_rssi, num_epochs=100)

    # ✅ 准备增强样本总数
    num_to_generate = 2000
    z = torch.randn(num_to_generate, 100).to(device)
    synthetic_rssi = rssi_gan(z).detach().cpu().numpy()
    synthetic_rssi = (synthetic_rssi + 1) / 2 * 105 - 105
    synthetic_rssi = np.clip(synthetic_rssi, -105, 0)

    synthetic_df = pd.DataFrame(synthetic_rssi, columns=train_data.filter(regex='^WAP').columns)
    synthetic_df['FLOOR'] = 4
    synthetic_df['BUILDINGID'] = 2  # ✅ 你数据中 Floor 4 所属 Building ID

    # ✅ 利用 KMeans 对 Floor 4 的原始样本位置进行空间划分
    num_clusters = 4
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)

    coords = floor4_data[['LONGITUDE', 'LATITUDE']].values
    cluster_labels = kmeans.fit_predict(coords)

    # 将每个 cluster 的经纬度采样赋值给 synthetic_df 子集
    samples_per_cluster = num_to_generate // num_clusters
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

    # 拼接所有聚类样本
    synthetic_df_final = pd.concat(synthetic_parts, ignore_index=True)

    # 合并到训练集
    train_data = pd.concat([train_data, synthetic_df_final], ignore_index=True)

    print(f"✅ Floor 4 样本增强后数量: {len(train_data[train_data['FLOOR'] == 4])}")

    # 标签修正（BUILDINGID/FLOOR 从 0 开始）
    for col in ['BUILDINGID', 'FLOOR']:
        # 计算训练集的最小值
        train_min = train_data[col].min()
        # 调整训练集标签
        train_data[col] -= train_min
        # 验证集使用相同的调整值
        val_data[col] -= train_min

    # 验证标签是否从0开始
    assert train_data['BUILDINGID'].min() == 0
    assert train_data['FLOOR'].min() == 0
    assert val_data['BUILDINGID'].min() == 0
    assert val_data['FLOOR'].min() == 0

    print("训练集标签范围:")
    print(f"Building ID范围: {train_data['BUILDINGID'].min()} - {train_data['BUILDINGID'].max()}")
    print(f"Floor ID范围: {train_data['FLOOR'].min()} - {train_data['FLOOR'].max()}")

    print("\n验证集标签范围:")
    print(f"Building ID范围: {val_data['BUILDINGID'].min()} - {val_data['BUILDINGID'].max()}")
    print(f"Floor ID范围: {val_data['FLOOR'].min()} - {val_data['FLOOR'].max()}")

    num_buildings = int(train_data['BUILDINGID'].max()) + 1
    num_floors = int(train_data['FLOOR'].max()) + 1

    print(f"模型配置:")
    print(f"建筑物数量: {num_buildings}")
    print(f"楼层数量: {num_floors}")

    # 计算训练集和验证集每栋每层的样本数
    train_building_floor_counts = train_data.groupby(['BUILDINGID', 'FLOOR']).size()
    val_building_floor_counts = val_data.groupby(['BUILDINGID', 'FLOOR']).size()

    # 原始坐标
    coord_train = train_data[['LONGITUDE', 'LATITUDE']].values
    coord_val = val_data[['LONGITUDE', 'LATITUDE']].values

    # 计算均值方差
    coord_mean = coord_train.mean(axis=0)
    coord_std = coord_train.std(axis=0)

    # 标准化（推荐）
    coord_train = (coord_train - coord_mean) / coord_std
    coord_val = (coord_val - coord_mean) / coord_std

    building_ids_train = train_data['BUILDINGID'].values
    floor_ids_train = train_data['FLOOR'].values
    building_ids_val = val_data['BUILDINGID'].values
    floor_ids_val = val_data['FLOOR'].values

    # 原CDAE代码替换为:
    print("Training DAE model...")
    dae = train_dae(
        features=train_data.filter(regex='^WAP').values,
        encoding_dim=128
    )

    print("Applying DAE to train/val data...")
    denoised_train_features = apply_dae(
        dae,
        features=train_data.filter(regex='^WAP').values
    )

    denoised_val_features = apply_dae(
        dae,
        features=val_data.filter(regex='^WAP').values
    )

    # 5. 创建数据集（传入去噪后的特征）
    train_dataset = WiFiDataset(
        features=denoised_train_features,
        labels=train_data['FLOOR'].values,
        building_ids=train_data['BUILDINGID'].values,
        coordinates=coord_train,
        seq_length=seq_length,
        is_train=True  # 标记训练集
    )

    val_dataset = WiFiDataset(
        features=denoised_val_features,
        labels=val_data['FLOOR'].values,
        building_ids=val_data['BUILDINGID'].values,
        coordinates=coord_val,
        seq_length=seq_length,
        imputer=train_dataset.imputer,  # 使用训练集的imputer
        scaler=train_dataset.scaler,  # 使用训练集的scaler
        is_train=False  # 标记验证集
    )

    if train_subset_ratio < 1.0:
        subset_size = int(train_subset_ratio * len(train_dataset))
        train_dataset = torch.utils.data.Subset(train_dataset, range(subset_size))

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 计算 floor_weights 时，正确处理 Subset 情况
    if isinstance(train_loader.dataset, Subset):
        # 如果是 Subset，通过 .dataset 访问原始数据集的 labels
        floor_counts = np.bincount(train_loader.dataset.dataset.labels)
    else:
        # 否则直接访问 labels
        floor_counts = np.bincount(train_loader.dataset.labels)

    return train_loader, val_loader, num_buildings, num_floors, floor_counts, coord_mean, coord_std


# 主函数
def main():
    # 初始化配置
    model_config = {
        'input_dim': 520,
        'spatial_feature_dim': 256,  # 对应cnn_channels
        'temporal_hidden_dim': 512,  # 对应gru_hidden_size
        'temporal_feature_dim': 256,  # GRU输出维度保持与空间特征相同
        'eca_channels': 512,
        'num_classes': None,
        'fc1_dropout': 0.5,  # 新增参数
        'dae_noise_std': 0.02  # 新增参数
    }

    # 设置路径
    # 在主函数中修改这两行
    TRAIN_PATH = '/kaggle/input/ujiindoorloc/TrainingData.csv'
    VAL_PATH = '/kaggle/input/ujiindoorloc/ValidationData.csv'
    # 加载并分层划分数据
    train_loader, val_loader, num_buildings, num_floors, floor_counts, coord_mean, coord_std = load_and_preprocess_data(
        train_data_path=TRAIN_PATH,
        val_data_path=VAL_PATH,
        train_subset_ratio=1.0,
    )
    train_dataset_size = len(train_loader.dataset)
    print(f"训练集样本数: {train_dataset_size}")

    # 更新模型配置
    model_config['num_classes'] = num_buildings + num_floors
    model_config['num_buildings'] = num_buildings
    model_config['num_floors'] = num_floors
    print(f"模型配置: 建筑数量={num_buildings}, 楼层数量={num_floors}")

    # 初始化模型
    model = WiFiLocalizationModel(model_config)

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.7, patience=3
    # )

    num_epochs = 60

    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)

    print(f"Scheduler is of type: {type(scheduler)}")

    # 计算楼层权重
    floor_weights = 1.0 / (floor_counts + 1e-7)
    floor_weights = torch.tensor(floor_weights, dtype=torch.float32).to(device)

    # 使用 FocalLoss 或 CrossEntropyLoss
    # floor_loss_fn = FocalLoss(alpha=floor_weights, gamma=2)
    floor_loss_fn = nn.CrossEntropyLoss(weight=floor_weights, label_smoothing=0.05)

    # 启动训练
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        floor_loss_fn=floor_loss_fn,
        criterion=criterion,
        scheduler=scheduler,
        optimizer=optimizer,
        num_epochs=60,
        patience=20,
        coord_mean=coord_mean,
        coord_std=coord_std,
        flood_level=0.3,  # 设置 Flooding 阈值（可调整）
        device=device  # 添加这行
    )

    logger.info("✅ Training completed!")


if __name__ == '__main__':
    main()
















