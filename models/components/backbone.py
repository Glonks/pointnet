import torch
import torch.nn as nn

from .transform_net import TransformNet


class PointNetBackbone(nn.Module):
    def __init__(self, point_dim):
        super().__init__()

        self.input_transform_net = TransformNet(point_dim)

        self.shared_mlp_1 = nn.Sequential(
            nn.Conv1d(point_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.feature_transform_net = TransformNet(point_dim=64)

        self.shared_mlp_2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

    def forward(self, x):
        input_transform = self.input_transform_net(x)
        x = input_transform @ x

        x = self.shared_mlp_1(x)

        feature_transform = self.feature_transform_net(x)
        local_features = feature_transform @ x

        x = self.shared_mlp_2(local_features)

        global_features = torch.max(x, dim=-1)[0]

        return local_features, global_features, feature_transform
