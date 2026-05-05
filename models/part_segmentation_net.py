import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import TransformNet


class SharedMLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.shared_mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.shared_mlp(x)


class PartSegmentationBackbone(nn.Module):
    def __init__(self, point_dim):
        super().__init__()

        self.input_transform_net = TransformNet(point_dim)

        self.shared_mlp_1 = nn.ModuleList([
            SharedMLPBlock(point_dim, 64),
            SharedMLPBlock(64, 128),
            SharedMLPBlock(128, 128),
        ])

        self.feature_transform_net = TransformNet(point_dim=128)

        self.shared_mlp_2 = nn.ModuleList([
            SharedMLPBlock(128, 512),
            SharedMLPBlock(512, 2048),
        ])

    def forward(self, x):
        intermediate_features = []

        input_transform = self.input_transform_net(x)
        x = input_transform @ x

        x = self.shared_mlp_1[0](x)
        intermediate_features.append(x)

        for module in self.shared_mlp_1[1:]:
            x = module(intermediate_features[-1])
            intermediate_features.append(x)

        feature_transform = self.feature_transform_net(x)
        intermediate_features.append(feature_transform @ x)

        for module in self.shared_mlp_2[:-1]:
            x = module(intermediate_features[-1])
            intermediate_features.append(x)

        x = self.shared_mlp_2[-1](x)

        global_features = torch.max(x, dim=-1)[0]

        return intermediate_features, global_features, feature_transform


class PartSegmentationNet(nn.Module):
    def __init__(self, point_dim, num_classes, device):
        super().__init__()

        self.device = device

        self.backbone = PartSegmentationBackbone(point_dim)

        # segmentation head
        self.shared_mlp = nn.Sequential(
            SharedMLPBlock(3024, 256),
            SharedMLPBlock(256, 256),
            SharedMLPBlock(256, 128),
            nn.Conv1d(128, num_classes, kernel_size=1)
        )

        self.to(device)

    def forward(self, points, category):
        intermediate_features, global_features, feature_transform = self.backbone(points)

        global_features = (
            global_features
            .unsqueeze(-1)
            .expand(-1, -1, intermediate_features[0].shape[-1])  # expand to number of points
        )
        category = (
            category
            .unsqueeze(-1)
            .expand(-1, -1, intermediate_features[0].shape[-1])
        )

        fused_features = torch.cat([
            *intermediate_features,
            global_features,
            category
        ], dim=1)
        
        logits = self.shared_mlp(fused_features)

        if self.training:
            return logits, feature_transform

        return logits
