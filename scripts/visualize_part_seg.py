import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tyro
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

from dataclasses import dataclass
from torch.utils.data import DataLoader

from datasets import ShapeNetPart
from models import PartSegmentationNet


PART_COLORS = np.array([
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8',  # Airplane
    '#f58231', '#911eb4',  # Bag
    '#42d4f4', '#f032e6',  # Cap
    '#bfef45', '#fabed4', '#469990', '#dcbeff',  # Car
    '#9A6324', '#fffac8', '#800000', '#aaffc3',  # Chair
    '#808000', '#ffd8b1', '#000075',  # Earphone
    '#a9a9a9', '#ffffff', '#000000',  # Guitar
    '#e6194b', '#3cb44b',  # Knife
    '#ffe119', '#4363d8', '#f58231', '#911eb4',  # Lamp
    '#42d4f4', '#f032e6',  # Laptop
    '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8',  # Motorbike
    '#800000', '#aaffc3',  # Mug
    '#808000', '#ffd8b1', '#000075',  # Pistol
    '#a9a9a9', '#ffffff', '#000000',  # Rocket
    '#e6194b', '#3cb44b', '#ffe119',  # Skateboard
    '#4363d8', '#f58231', '#911eb4',  # Table
])


def save_gif(points, labels, predictions, category_name, valid_parts, save_path):
    points = points.T[:, :3]
    points = points - np.mean(points, axis=0)
    points = points / np.max(np.linalg.norm(points, axis=1))

    figure = plt.figure(figsize=(10, 5))
    axis_gt   = figure.add_subplot(121, projection='3d')
    axis_pred = figure.add_subplot(122, projection='3d')

    def scatter_by_axis(axis, point_labels, title):
        scatter = axis.scatter(
            points[:, 0], points[:, 2], points[:, 1],
            s=5,
            c=PART_COLORS[point_labels]
        )
        axis.set_title(title)
        axis.set_axis_off()
        axis.set_box_aspect([1, 1, 1])
        return scatter

    scatter_gt = scatter_by_axis(
        axis_gt,
        labels,
        f'Ground Truth: {category_name}'
    )
    scatter_pred = scatter_by_axis(
        axis_pred,
        predictions,
        f'Prediction: {category_name}'
    )

    patches = [
        mpatches.Patch(color=PART_COLORS[part], label=f'Part {part}')  # type: ignore
        for part in valid_parts
    ]
    figure.legend(
        handles=patches,
        loc='lower center',
        ncol=len(valid_parts),
        fontsize=8,
        framealpha=0.5
    )

    def update(frame):
        axis_gt.view_init(elev=20, azim=frame)
        axis_pred.view_init(elev=20, azim=frame)
        return scatter_gt, scatter_pred

    anim = animation.FuncAnimation(
        figure,
        update,  # type: ignore
        frames=np.linspace(0, 360, 60),
        interval=100
    )

    anim.save(save_path, writer='pillow', fps=10, dpi=150)

    plt.close()


@dataclass
class Config:
    name:            str = 'part_seg'
    checkpoint_path: str = 'part_seg_best.pt'
    num_points:      int = 2048
    batch_size:      int = 4
    device:          str = 'cuda'


def main(config: Config):
    print(config)

    dataset = ShapeNetPart(num_points=config.num_points, split='test')
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    checkpoint = torch.load(
        config.checkpoint_path,
        map_location=config.device,
        weights_only=False
    )

    model = PartSegmentationNet(
        point_dim=6,
        num_classes=ShapeNetPart.NUM_PARTS,
        device=config.device
    )
    model.load_state_dict(checkpoint['model'])
    model.eval()

    points, category, labels = next(iter(loader))

    points = points.to(model.device)
    category = category.float().to(model.device)

    with torch.no_grad():
        logits = model(points, category)
        predictions = logits.argmax(dim=1)

    points = points.cpu().numpy()
    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()
    category_index = category.argmax(dim=-1).cpu().numpy()

    result_root = os.path.join('results', config.name)
    os.makedirs(result_root, exist_ok=True)

    for i in range(points.shape[0]):
        category_name = ShapeNetPart.CATEGORIES[category_index[i]]
        valid_parts = ShapeNetPart.CATEGORY_PART_MAPPING[category_name]

        save_gif(
            points[i],
            labels=labels[i],
            predictions=predictions[i],
            category_name=category_name,
            valid_parts=valid_parts,
            save_path=os.path.join(result_root, f'{i}_{category_name}.gif')
        )

    print('Results generated')


if __name__ == '__main__':
    config = tyro.cli(Config)
    main(config)
