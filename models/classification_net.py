import torch
import torch.nn as nn

from .components import PointNetBackbone


class ClassificationNet(nn.Module):
    def __init__(self, point_dim, num_classes, device):
        super().__init__()

        self.device = device

        self.backbone = PointNetBackbone(point_dim)

        # classification head
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        self.to(device)

    def forward(self, x):
        _, global_features, feature_transform = self.backbone(x)

        logits = self.mlp(global_features)

        if self.training:
            return logits, feature_transform

        return logits
