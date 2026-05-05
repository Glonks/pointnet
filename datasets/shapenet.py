import os
import json
import torch
import zipfile
import numpy as np
import torch.nn.functional as F

from torch.utils.data import Dataset
from huggingface_hub import snapshot_download


class ShapeNetPart(Dataset):
    REPO_ID        = 'wangps/shapenet_segmentation'
    ROOT           = 'data/shapenet'
    DATA_DIR       = 'shapenetcore_partanno_segmentation_benchmark_v0_normal'
    ZIP_NAME       = 'shapenetcore_partanno_segmentation_benchmark_v0_normal.zip'
    NUM_MAX_POINTS = 2048
    CATEGORY_PART_MAPPING = {
        'Airplane':   [0, 1, 2, 3],
        'Bag':        [4, 5],
        'Cap':        [6, 7],
        'Car':        [8, 9, 10, 11],
        'Chair':      [12, 13, 14, 15],
        'Earphone':   [16, 17, 18],
        'Guitar':     [19, 20, 21],
        'Knife':      [22, 23],
        'Lamp':       [24, 25, 26, 27],
        'Laptop':     [28, 29],
        'Motorbike':  [30, 31, 32, 33, 34, 35],
        'Mug':        [36, 37],
        'Pistol':     [38, 39, 40],
        'Rocket':     [41, 42, 43],
        'Skateboard': [44, 45, 46],
        'Table':      [47, 48, 49],
    }
    CATEGORIES = list(CATEGORY_PART_MAPPING.keys())
    NUM_CATEGORIES = len(CATEGORIES)
    NUM_PARTS = 50

    def __init__(self, num_points, split):
        if split not in {'train', 'test', 'val'}:
            raise ValueError('Unknown split')
        
        if num_points > self.NUM_MAX_POINTS:
            raise ValueError(
                f'Cannot provide more points ({num_points}) than the number of '
                f'points this dataset was prepared with ({self.NUM_MAX_POINTS})'
            )

        self.num_points = num_points
        self.split = split

        self._download()
        self._prepare()

        npz_path = os.path.join(self.ROOT, f'{self.split}.npz')
        npz = np.load(npz_path)

        self.points = npz['points']
        self.labels = npz['labels']
        self.categories = npz['categories']

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

        wordnet_mapping = self._get_wordnet_mapping()
        category_ids = {
            category: i
            for i, category
            in enumerate(self.CATEGORY_PART_MAPPING.keys())
        }

        for split in ['train', 'test', 'val']:
            split_files = self._gather_split(split)

            all_points = []
            all_labels = []
            all_categories = []

            for split_file in split_files:
                synset_id = split_file.split('/')[3]
                category_name = wordnet_mapping[synset_id]
                category_id = category_ids[category_name]

                data = np.loadtxt(split_file)

                _points = data[:, :-1]
                _labels = data[:, -1].astype(np.int64)

                # Sample down/up to NUM_POINTS to make matrices rectangular for storage
                if _points.shape[0] >= self.NUM_MAX_POINTS:
                    choice = np.random.choice(_points.shape[0], self.NUM_MAX_POINTS, replace=False)
                else:
                    choice = np.random.choice(_points.shape[0], self.NUM_MAX_POINTS, replace=True)

                _points = _points[choice]
                _labels = _labels[choice]

                all_points.append(_points)
                all_labels.append(_labels)
                all_categories.append(category_id)

            all_points = np.stack(all_points, axis=0)
            all_labels = np.stack(all_labels, axis=0)
            all_categories = np.array(all_categories, dtype=np.int64)

            np.savez(
                os.path.join(self.ROOT, f'{split}.npz'),
                points=all_points,
                labels=all_labels,
                categories=all_categories
            )

        open(prepared, 'w').close()

        print('Dataset prepared')
    
    def _get_wordnet_mapping(self):
        data_root = os.path.join(self.ROOT, self.DATA_DIR)

        wordnet_mapping_path = os.path.join(data_root, 'synsetoffset2category.txt')
        with open(wordnet_mapping_path, 'r') as file:
            wordnet_mapping = dict([
                line.split()[::-1]
                for line
                in file.read().splitlines()
            ])
        
        return wordnet_mapping

    def _gather_split(self, split):
        data_root = os.path.join(self.ROOT, self.DATA_DIR)

        split_info_file = os.path.join(data_root, 'train_test_split', f'shuffled_{split}_file_list.json')
        with open(split_info_file, 'r') as file:
            split_files = [
                os.path.join(data_root, f'{line.split('/', 1)[-1]}.txt')
                for line
                in json.load(file)
            ]
        
        return split_files

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        points = self.points[index]
        labels = self.labels[index]
        category = self.categories[index]

        choice = np.random.choice(points.shape[0], self.num_points, replace=False)
        points = points[choice]
        labels = labels[choice]

        points = torch.from_numpy(points).float().T
        labels = torch.from_numpy(labels).long()

        category = torch.tensor(category, dtype=torch.long)
        category = F.one_hot(category, num_classes=self.NUM_CATEGORIES)

        return points, category, labels
