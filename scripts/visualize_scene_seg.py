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

from datasets import S3DIS
from models import SceneSegmentationNet


CLASS_COLORS = {
    'ceiling':  'lightgray',
    'floor':    'saddlebrown',
    'wall':     'tan',
    'beam':     'orange',
    'column':   'mediumpurple',
    'window':   'deepskyblue',
    'door':     'sienna',
    'table':    'forestgreen',
    'chair':    'tomato',
    'sofa':     'gold',
    'bookcase': 'steelblue',
    'board':    'hotpink',
    'clutter':  'dimgray',
}
COLOR_LUT = np.array([
    CLASS_COLORS[class_name]
    for class_name
    in S3DIS.CLASSES
])


def save_gif(points, labels, predictions, save_path):
    points = points.T[:, :3] # (N, 9)

    figure = plt.figure()
    axis_gt = figure.add_subplot(121, projection='3d')
    axis_pred = figure.add_subplot(122, projection='3d')

    points = points - np.mean(points, axis=0)

    scale = np.max(np.linalg.norm(points, axis=1))
    points = points / scale

    def scatter_by_axis(axis, colors, title):
        scatter = axis.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=5,
            c=colors
        )

        axis.set_title(title)

        axis.set_axis_off()

        axis.set_box_aspect([1, 1, 1])

        return scatter

    scatter_gt = scatter_by_axis(
        axis_gt,
        COLOR_LUT[labels],
        'Ground Truth'
    )
    scatter_pred = scatter_by_axis(
        axis_pred,
        COLOR_LUT[predictions],
        'Prediction'
    )

    patches = [
        mpatches.Patch(color=CLASS_COLORS[class_name], label=class_name)
        for class_name
        in S3DIS.CLASSES
    ]
    figure.legend(
        handles=patches,
        loc='lower center',
        ncol=7,
        fontsize=6,
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

    anim.save(save_path, writer='pillow', fps=10, dpi=300)

    plt.close()


@dataclass
class Config:
    name:            str = 'scene_seg'
    checkpoint_path: str = 'scene_seg_best.pt'
    num_points:      int = 4096
    batch_size:      int = 4
    device:          str = 'cuda'


def main(config: Config):
    print(config)

    dataset = S3DIS(num_points=config.num_points, split='test')
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    checkpoint = torch.load(
        config.checkpoint_path,
        map_location=config.device,
        weights_only=False
    )

    model = SceneSegmentationNet(
        point_dim=9,
        num_classes=S3DIS.NUM_CLASSES,
        device=config.device
    )
    model.load_state_dict(checkpoint['model'])
    model.eval()

    points, labels = next(iter(loader))
    points = points.to(model.device)

    logits = model(points)
    predictions = logits.argmax(dim=1)

    points = points.cpu()
    labels = labels.cpu()
    predictions = predictions.cpu()

    result_root = os.path.join('results', config.name)
    os.makedirs(result_root, exist_ok=True)

    for i in range(points.shape[0]):
        save_gif(
            points[i].numpy(),
            labels=labels[i].numpy(),
            predictions=predictions[i].numpy(),
            save_path=os.path.join(result_root, f'{i}.gif')
        )
    
    print('Results generated')


if __name__ == '__main__':
    config = tyro.cli(Config)

    main(config)
