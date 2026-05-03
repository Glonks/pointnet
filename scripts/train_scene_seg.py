import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tyro
import torch
import numpy as np
import torch.nn.functional as F

from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import Literal

from datasets import S3DIS
from models import SceneSegmentationNet
from logger import Logger


@dataclass
class Config:
    name:          str                    = 'scene_seg'
    num_points:    int                    = 4096
    num_epochs:    int                    = 50
    batch_size:    int                    = 24
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
        'reg': [],
        'total': []
    }
    for i, (points, labels) in enumerate(loader):
        points = points.to(model.device)
        labels = labels.to(model.device)

        optimizer.zero_grad()

        logits, feature_transform = model(points)

        class_loss = F.cross_entropy(logits, labels)
        reg_loss = compute_reg_loss(feature_transform)

        loss = class_loss + reg_lambda * reg_loss

        loss.backward()
        optimizer.step()

        losses['class'].append(class_loss.item())
        losses['reg'].append(reg_loss.item())
        losses['total'].append(loss.item())

        if (i % 20 == 0) or (i == len(loader) - 1):
            logger.log({
                'Epoch': epoch,
                'Iteration': i,
                'Total Loss': loss.item(),
                'Class Loss': class_loss.item(),
                'Reg. Loss': reg_loss.item(),
            }, overwrite=True)

    scheduler.step()

    return {
        loss_type: sum(loss_val) / len(loss_val)
        for loss_type, loss_val
        in losses.items()
    }


def eval_one_epoch(model, loader, epoch, logger):
    model.eval()

    correct = 0
    total = 0
    intersection = np.zeros(S3DIS.NUM_CLASSES)
    union = np.zeros(S3DIS.NUM_CLASSES)
    class_correct = np.zeros(S3DIS.NUM_CLASSES)
    class_total = np.zeros(S3DIS.NUM_CLASSES)

    with torch.no_grad():
        for points, labels in loader:
            points = points.to(model.device)
            labels = labels.to(model.device)

            logits = model(points)
            predictions = logits.argmax(dim=1)

            correct += (
                (predictions == labels)
                .sum()
                .item()
            )
            total += labels.numel()

            for cls in range(S3DIS.NUM_CLASSES):
                predicted_cls = (predictions == cls)
                label_cls = (labels == cls)
                intersection[cls] += (predicted_cls & label_cls).sum().item()
                union[cls] += (predicted_cls | label_cls).sum().item()
                class_correct[cls] += (predictions[label_cls] == labels[label_cls]).sum().item()
                class_total[cls] += label_cls.sum().item()

    accuracy = correct / total
    iou = intersection / np.maximum(union, 1)
    miou = iou.mean()
    per_class_accuracy = class_correct / np.maximum(class_total, 1)
    mean_class_accuracy = per_class_accuracy.mean()

    return accuracy, mean_class_accuracy, miou


def main(config: Config):
    print(config)

    logger = Logger('point_scene_seg', config)

    train_dataset = S3DIS(num_points=config.num_points, split='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_dataset = S3DIS(num_points=config.num_points, split='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = SceneSegmentationNet(
        point_dim=9,
        num_classes=S3DIS.NUM_CLASSES,
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
        gamma=0.7
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

        accuracy, mean_class_accuracy, miou = eval_one_epoch(
            model,
            test_loader,
            epoch,
            logger
        )

        logger.log({
            'Epoch':                    epoch,
            'Train/Avg. Total Loss':    average_losses['total'],
            'Train/Avg. Class Loss':    average_losses['class'],
            'Train/Avg. Reg. Loss':     average_losses['reg'],
            'Test/Accuracy':            accuracy,
            'Test/Mean Class Accuracy': mean_class_accuracy,
            'Test/mIoU':                miou,
        })

        if miou > best_miou:
            best_miou = miou

            print('New best model - Saving')

            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'miou': best_miou
            }, f'{config.name}_best.pt')


if __name__ == '__main__':
    config = tyro.cli(Config)

    main(config)
