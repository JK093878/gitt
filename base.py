# models/base_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler

class RSSIGenerator(nn.Module):
    def __init__(self, noise_dim=100, output_dim=520):
        super(RSSIGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


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


def train_dae(features, encoding_dim=128, noise_std=0, epochs=30, device='cuda'):
    """训练DAE模型"""
    features = features.copy()
    features[features == 100] = -105

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    features_tensor = torch.FloatTensor(features).to(device)
    input_dim = features.shape[1]
    dae = DAE(input_dim=input_dim, encoding_dim=encoding_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(dae.parameters(), lr=1e-3)

    dae.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        noise = torch.randn_like(features_tensor) * noise_std
        noisy_input = torch.clamp(features_tensor + noise, 0, 1)
        outputs = dae(noisy_input)
        loss = criterion(outputs, features_tensor)
        loss.backward()
        optimizer.step()

    return dae, scaler


def apply_dae(dae, features, device='cuda'):
    """应用DAE去噪"""
    features = features.copy()
    features[features == 100] = -104
    features = (features + 104) / 104

    features_tensor = torch.FloatTensor(features).to(device)

    dae.eval()
    with torch.no_grad():
        outputs = dae(features_tensor)

    denoised = outputs.cpu().numpy() * 104 - 104
    return denoised

class ECAModule(nn.Module):
    def __init__(self, channels=0, gamma=2, b=1):
        super(ECAModule, self).__init__()
        self.channels = channels
        self.gamma = gamma
        self.b = b

        t = int(abs((np.log2(self.channels) + self.b) / self.gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

    def forward(self, x):
        batch_size, seq_length, channels = x.size()
        x_reshaped = x.reshape(-1, channels)
        x_channel = x_reshaped.unsqueeze(1)

        y = self.conv(x_channel)
        y = torch.sigmoid(y).squeeze(1)
        y = y.reshape(batch_size, seq_length, channels)

        return x * y

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

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        spatial_features = []

        for t in range(seq_len):
            x_t = x[:, t, :].unsqueeze(1)
            out = self.relu(self.bn1(self.conv1(x_t)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.relu(self.bn3(self.conv3(out)))
            out = self.pool(out).squeeze(-1)
            spatial_features.append(out.unsqueeze(1))

        return torch.cat(spatial_features, dim=1)

class TemporalFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, feature_dim):
        super(TemporalFeatureExtractor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        temporal_features = self.relu(self.fc(gru_out))
        return temporal_features