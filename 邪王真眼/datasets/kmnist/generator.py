import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from typing import Any, Dict
from torch.utils.data import Dataset


class KMNIST(Dataset):
    def __init__(self, split: str, cfg: Dict[str, Any]):
        if split not in ["train", "val"]:
            raise ValueError(f"Unknown split '{split}', expected 'train' or 'val'")
        
        data_root = cfg.get('data_root')
        transform = transforms.ToTensor()

        is_train = (split == "train")
        self.dataset = datasets.KMNIST(
            root=data_root,
            train=is_train,
            download=True,
            transform=transform
        )

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        return {
            'x': img,
            'label': torch.tensor(label, dtype=torch.long)
        }
