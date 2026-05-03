import os
import h5py
import torch
import zipfile
import numpy as np

from torch.utils.data import Dataset
from huggingface_hub import snapshot_download


class ModelNet40(Dataset):
    REPO_ID  = 'Msun/modelnet40'
    ROOT     = 'data/modelnet40'
    DATA_DIR = 'modelnet40_ply_hdf5_2048'
    ZIP_NAME = 'modelnet40_ply_hdf5_2048.zip'
    CLASSES = [
        "airplane",
        "bathtub",
        "bed",
        "bench",
        "bookshelf",
        "bottle",
        "bowl",
        "car",
        "chair",
        "cone",
        "cup",
        "curtain",
        "desk",
        "door",
        "dresser",
        "flower_pot",
        "glass_box",
        "guitar",
        "keyboard",
        "lamp",
        "laptop",
        "mantel",
        "monitor",
        "night_stand",
        "person",
        "piano",
        "plant",
        "radio",
        "range_hood",
        "sink",
        "sofa",
        "stairs",
        "stool",
        "table",
        "tent",
        "toilet",
        "tv_stand",
        "vase",
        "wardrobe",
        "xbox",
    ]
    NUM_CLASSES = len(CLASSES)

    def __init__(self, num_points, split, subsample=True):
        if split not in {'train', 'test'}:
            raise ValueError('Unknown split')

        self.num_points = num_points
        self.split = split
        self.subsample = subsample

        self._download()
        self._prepare()

        npz_path = os.path.join(self.ROOT, f'{self.split}.npz')
        npz = np.load(npz_path)

        self.points = npz['points']
        self.labels = npz['labels']

    def _download(self):
        if os.path.exists(self.ROOT):
            return

        snapshot_download(
            repo_id=self.REPO_ID,
            repo_type='dataset',
            local_dir=self.ROOT,
        )
    
    def _prepare(self):
        prepared = os.path.join(self.ROOT, '.prepared')
        if os.path.exists(prepared):
            return

        print('Preparing dataset')

        zip_path = os.path.join(self.ROOT, self.ZIP_NAME)
        with zipfile.ZipFile(zip_path, 'r') as file:
            file.extractall(self.ROOT)
        
        for split in ['train', 'test']:
            h5_files = self._gather_split(split)
            
            all_points = []
            all_labels = []

            for h5_file in h5_files:
                with h5py.File(h5_file, 'r') as file:
                    all_points.append(file['data'][:])  # type: ignore
                    all_labels.append(file['label'][:])  # type: ignore

            points = np.concatenate(all_points, axis=0)
            labels = np.concatenate(all_labels, axis=0).squeeze()
                
            save_path = os.path.join(self.ROOT, f'{split}.npz')
            np.savez(save_path, points=points, labels=labels)

        # touch
        open(prepared, 'w').close()

        print('Dataset prepared')
    
    def _gather_split(self, split):
        # Read split file based on split and produce list of associated split files
        split_list_file = os.path.join(self.ROOT, self.DATA_DIR, f'{split}_files.txt')
        with open(split_list_file, 'r') as file:
            h5_files = [
                os.path.join(self.ROOT, self.DATA_DIR, line.rsplit('/', 1)[-1])
                for line
                in file.read().splitlines()
            ]

        return h5_files

    def __len__(self):
        return len(self.labels)

    def _rotate(self, points):
        theta = np.random.uniform(0, 2 * np.pi)
        cos, sin = np.cos(theta), np.sin(theta)

        # Y up
        R = np.array([
            [ cos, 0, sin],
            [  0,  1,   0],
            [-sin, 0, cos]
        ], dtype=float)

        return points @ R.T

    def _jitter(self, points):
        return (
            points +
            np.random.normal(0, 0.02, points.shape).astype(float)
        )

    def __getitem__(self, index):
        points = self.points[index]
        label = self.labels[index]

        if self.subsample:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[choice].copy()

        if self.split == 'train':
            points = self._rotate(points)
            points = self._jitter(points)

        points = torch.from_numpy(points).float().T
        label  = torch.tensor(label).long()

        return points, label
