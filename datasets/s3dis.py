import os
import h5py
import torch
import zipfile
import numpy as np

from torch.utils.data import Dataset
from huggingface_hub import snapshot_download


class S3DIS(Dataset):
    REPO_ID   = 'cminst/S3DIS'
    ROOT      = 'data/s3dis'
    DATA_DIR  = 'indoor3d_sem_seg_hdf5_data'
    ZIP_NAME  = 'indoor3d_sem_seg_hdf5_data.zip'
    TEST_AREA = 6
    CLASSES = [
        'ceiling',
        'floor',
        'wall',
        'beam',
        'column',
        'window',
        'door',
        'table',
        'chair',
        'sofa',
        'bookcase',
        'board',
        'clutter',
    ]
    NUM_CLASSES = len(CLASSES)

    def __init__(self, num_points, split):
        if split not in {'train', 'test'}:
            raise ValueError('Unknown split')

        self.num_points = num_points
        self.split = split

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

        all_h5_files, room_names = self._gather_h5_files_and_room_names()

        all_points = []
        all_labels = []

        for h5_path in all_h5_files:
            with h5py.File(h5_path, 'r') as f:
                all_points.append(f['data'][:])  # type: ignore
                all_labels.append(f['label'][:])  # type: ignore
        
        all_points = np.concatenate(all_points, axis=0)  # (23585, 4096, 9)
        all_labels = np.concatenate(all_labels, axis=0)  # (23585, 4096)

        test_mask  = np.array([
            f'Area_{self.TEST_AREA}' in room_name
            for room_name
            in room_names
        ])
        train_mask = ~test_mask

        for split, mask in [('train', train_mask), ('test', test_mask)]:
            np.savez(
                os.path.join(self.ROOT, f'{split}.npz'),
                points=all_points[mask],
                labels=all_labels[mask],
            )

        open(prepared, 'w').close()

        print('Dataset prepared')
    
    def _gather_h5_files_and_room_names(self):
        data_root = os.path.join(self.ROOT, self.DATA_DIR)

        all_files_path = os.path.join(data_root, 'all_files.txt')
        with open(all_files_path, 'r') as file:
            all_h5_files = [
                os.path.join(data_root, line.split('/')[-1])
                for line
                in file.read().splitlines()
            ]
        
        room_filelist_path = os.path.join(data_root, 'room_filelist.txt')
        with open(room_filelist_path, 'r') as file:
            room_names = file.read().splitlines()
        
        return all_h5_files, room_names

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        points = self.points[index]
        labels = self.labels[index]

        choice = np.random.choice(points.shape[0], self.num_points, replace=False)
        points = points[choice]
        labels = labels[choice]

        points = torch.from_numpy(points).float().T
        labels = torch.from_numpy(labels).long()

        return points, labels
