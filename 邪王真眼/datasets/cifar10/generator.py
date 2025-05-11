import os
import torch
import numpy as np
import torchvision.transforms as transforms

from typing import Any, Dict
from torch.utils.data import Dataset


class CIFAR10Dataset(Dataset):
    def __init__(self, split: str, cfg: Dict[str, Any]):
        self.data_dir = os.path.join(cfg.get("data_root"), "cifar-10-batches-py")
        self.data = []
        self.labels = []

        batch_files = [f"data_batch_{i}" for i in range(1, 6)] if split == "train" else ["test_batch"]

        self.transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        for batch in batch_files:
            batch_path = os.path.join(self.data_dir, batch)
            batch_data = self.unpickle(batch_path)
            self.data.append(batch_data[b'data'])  # shape: (10000, 3072)
            self.labels += batch_data[b'labels']  # shape: (10000,)

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)  # shape: (N, 3, 32, 32)
    
    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        return data_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]  # shape: (3, 32, 32)
        label = self.labels[idx]

        image = torch.tensor(image, dtype=torch.float32) / 255.0  # 归一化到 [0, 1]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        image = self.transform(image)

        return {'x': image, 'label': label}
