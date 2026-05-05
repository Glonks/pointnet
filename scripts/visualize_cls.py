import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tyro
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dataclasses import dataclass
from torch.utils.data import DataLoader

from datasets import ModelNet40
from models import ClassificationNet


@dataclass
class Config:
    name:            str = 'obj_class'
    checkpoint_path: str = 'obj_class_best.pt'
    num_points:      int = 1024
    batch_size:      int = 4
    device:          str = 'cuda'


def save_gif(points, label, prediction, save_path):
    points = points.numpy().T  # (N, 3)
    points = points - np.mean(points, axis=0)
    points = points / np.max(np.linalg.norm(points, axis=1))

    figure = plt.figure()
    axis = figure.add_subplot(111, projection='3d')

    color = 'steelblue' if label == prediction else 'red'

    # Y Up
    scatter = axis.scatter(
        points[:, 0],
        points[:, 2],
        points[:, 1],
        s=5,
        c=color
    )
    axis.set_title(f'GT: {label} | Prediction: {prediction}')
    axis.set_axis_off()
    axis.set_box_aspect([1, 1, 1])

    def update(frame):
        axis.view_init(elev=20, azim=frame)
        return scatter

    anim = animation.FuncAnimation(
        figure,
        update,  # type: ignore
        frames=np.linspace(0, 360, 60),
        interval=100
    )

    anim.save(save_path, writer='pillow', fps=10, dpi=300)

    plt.close(figure)


def main(config: Config):
    print(config)

    dataset = ModelNet40(num_points=config.num_points, split='test')
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    checkpoint = torch.load(config.checkpoint_path, map_location=config.device)

    model = ClassificationNet(
        point_dim=3,
        num_classes=ModelNet40.NUM_CLASSES,
        device=config.device
    )
    model.load_state_dict(checkpoint['model'])
    model.eval()

    points, labels = next(iter(loader))
    points = points.to(model.device)

    with torch.no_grad():
        logits = model(points)
        predictions = logits.argmax(dim=1)

    points = points.cpu()
    labels = labels.cpu()
    predictions = predictions.cpu()

    result_root = os.path.join('results', config.name)
    os.makedirs(result_root, exist_ok=True)

    for i in range(points.shape[0]):
        save_gif(
            points[i],
            label=ModelNet40.CLASSES[labels[i].item()],
            prediction=ModelNet40.CLASSES[predictions[i].item()],
            save_path=os.path.join(result_root, f'{i}.gif')
        )
    
    print('Results generated')


if __name__ == '__main__':
    config = tyro.cli(Config)

    main(config)
