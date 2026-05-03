import torch
import torch.nn as nn

from .components import PointNetBackbone


class SceneSegmentationNet(nn.Module):
    def __init__(self, point_dim, num_classes, device):
        super().__init__()

        self.device = device

        self.backbone = PointNetBackbone(point_dim)

        # segmentation head
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(1088, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, num_classes, kernel_size=1)
        )

        self.to(device)

    def forward(self, x):
        local_features, global_features, feature_transform = self.backbone(x)

        global_features = (
            global_features
            .unsqueeze(-1)
            .expand(-1, -1, local_features.shape[-1])
        )

        fused_features = torch.cat([local_features, global_features], dim=1)
        
        logits = self.shared_mlp(fused_features)

        if self.training:
            return logits, feature_transform

        return logits
