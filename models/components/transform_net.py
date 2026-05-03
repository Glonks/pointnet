import torch
import torch.nn as nn


class TransformNet(nn.Module):
    def __init__(self, point_dim):
        super().__init__()

        self.point_dim = point_dim

        self.shared_mlp = nn.Sequential(
            nn.Conv1d(point_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.transform = nn.Linear(256, point_dim ** 2)
        with torch.no_grad():
            self.transform.weight.zero_()
            self.transform.bias.copy_(torch.eye(point_dim).flatten())

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.shared_mlp(x)

        x = torch.max(x, dim=-1)[0]

        x = self.mlp(x)

        x = self.transform(x)

        x = x.reshape(batch_size, self.point_dim, self.point_dim)

        return x
