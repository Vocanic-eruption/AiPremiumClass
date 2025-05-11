import torch
import numpy as np

from typing import Any, Dict
from torch.utils.data import Dataset
from sklearn.datasets import fetch_olivetti_faces

from models.models import ModelType


class OlivettiFaces(Dataset):
    def __init__(self, split: str, cfg: Dict[str, Any]):
        if split not in ["train", "val"]:
            raise ValueError(f"Unknown split '{split}', expected 'train' or 'val'")

        dataset_path = cfg.get('data_root', './data')
        faces = fetch_olivetti_faces(data_home=dataset_path, shuffle=False)

        X = faces.images  # (400, 64, 64)
        y = faces.target  # (400,)

        mean = X.mean()
        std = X.std()
        X = (X - mean) / std

        if cfg.get('model') == ModelType.CNN:
            X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (400, 1, 64, 64)
        else:
            X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        y = torch.tensor(y, dtype=torch.long)

        train_idx = np.array([i * 10 + j for i in range(40) for j in range(7)])
        val_idx = np.array([i * 10 + j for i in range(40) for j in range(7, 10)])

        if split == "train":
            self.X = X[train_idx]
            self.y = y[train_idx]
        else:
            self.X = X[val_idx]
            self.y = y[val_idx]

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img = self.X[idx]
        label = self.y[idx]
        return {
            'x': img,
            'label': label
        }
