import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tyro
import torch
import numpy as np
import torch.nn.functional as F

from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import Literal

from datasets import ShapeNetPart
from models import PartSegmentationNet
from logger import Logger


@dataclass
class Config:
    name:          str                    = 'part_seg'
    num_points:    int                    = 2048
    num_epochs:    int                    = 200
    batch_size:    int                    = 32
    reg_lambda:    float                  = 1e-3
    learning_rate: float                  = 1e-3
    momentum:      float                  = 0.9
    optimizer:     Literal['adam', 'sgd'] = 'adam'
    device:        str                    = 'cuda'


def compute_reg_loss(T):
    B, K, _ = T.shape

    I = (
        torch.eye(K, device=T.device)
        .unsqueeze(0)
        .expand(B, -1, -1)
    )

    difference = I - T @ T.transpose(-1, -2)

    return (
        (difference ** 2)
        .sum(dim=(1, 2))
        .mean()
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    reg_lambda,
    epoch,
    logger
):
    model.train()

    losses = {
        'class': [],
        'reg':   [],
        'total': []
    }
    for i, (points, category, labels) in enumerate(loader):
        points = points.to(model.device)
        labels = labels.to(model.device)
        category = category.float().to(model.device)

        optimizer.zero_grad()

        logits, feature_transform = model(points, category)

        class_loss = F.cross_entropy(logits, labels)
        reg_loss = compute_reg_loss(feature_transform)

        loss = class_loss + reg_lambda * reg_loss

        loss.backward()
        optimizer.step()

        losses['class'].append(class_loss.item())
        losses['reg'] .append(reg_loss.item())
        losses['total'].append(loss.item())

        if (i % 20 == 0) or (i == len(loader) - 1):
            logger.log({
                'Epoch':      epoch,
                'Iteration':  i,
                'Total Loss': loss.item(),
                'Class Loss': class_loss.item(),
                'Reg. Loss':  reg_loss.item(),
            }, overwrite=True)

    scheduler.step()

    return {
        loss_type: sum(loss_val) / len(loss_val)
        for loss_type, loss_val
        in losses.items()
    }


def eval_one_epoch(model, loader, epoch, logger):
    model.eval()

    intersection = np.zeros(ShapeNetPart.NUM_PARTS)
    union = np.zeros(ShapeNetPart.NUM_PARTS)

    with torch.no_grad():
        for points, category, labels in loader:
            points = points.to(model.device)
            labels = labels.to(model.device)
            category = category.to(model.device)

            logits = model(points, category)
            predictions = logits.argmax(dim=1)

            category_index = category.argmax(dim=-1)

            # only evaluate parts valid for each sample's category
            for sample_index in range(points.shape[0]):
                valid_parts = ShapeNetPart.CATEGORY_PART_MAPPING[
                    ShapeNetPart.CATEGORIES[category_index[sample_index].item()]
                ]
                for part in valid_parts:
                    predicted_part  = (predictions[sample_index] == part)
                    label_part = (labels[sample_index] == part)

                    intersection[part] += (predicted_part & label_part).sum().item()
                    union[part]        += (predicted_part | label_part).sum().item()

    part_iou = intersection / np.maximum(union, 1)

    category_miou = [
        part_iou[parts].mean()
        for parts
        in ShapeNetPart.CATEGORY_PART_MAPPING.values()
    ]

    return np.mean(category_miou)


def main(config: Config):
    print(config)

    logger = Logger('pointnet_part_seg', config)

    train_dataset = ShapeNetPart(num_points=config.num_points, split='train')
    train_loader  = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_dataset = ShapeNetPart(num_points=config.num_points, split='val')
    val_loader  = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = PartSegmentationNet(
        point_dim=6,
        num_classes=ShapeNetPart.NUM_PARTS,
        device=config.device
    )

    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )

    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum
        )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.5
    )

    best_miou = 0
    for epoch in range(config.num_epochs):
        average_losses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            config.reg_lambda,
            epoch,
            logger
        )

        miou = eval_one_epoch(model, val_loader, epoch, logger)

        logger.log({
            'Epoch':                 epoch,
            'Train/Avg. Total Loss': average_losses['total'],
            'Train/Avg. Class Loss': average_losses['class'],
            'Train/Avg. Reg. Loss':  average_losses['reg'],
            'Val/mIoU':              miou,
        })

        if miou > best_miou:
            best_miou = miou

            print('New best model - Saving')

            torch.save({
                'epoch':     epoch,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'miou':      float(best_miou),
            }, f'{config.name}_best.pt')


if __name__ == '__main__':
    config = tyro.cli(Config)

    main(config)
