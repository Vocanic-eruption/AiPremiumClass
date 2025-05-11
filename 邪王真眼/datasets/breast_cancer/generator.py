import torch

from typing import Any, Dict
from torch.utils.data import Dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class BreastCancer(Dataset):
    def __init__(self, split: str, _: Dict[str, Any]):
        if split not in ["train", "val"]:
            raise ValueError(f"Unknown split '{split}', expected one of ['train', 'val']")

        data = load_breast_cancer()
        X = data.data
        y = data.target
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42,
            stratify=y
        )
        
        if split == "train":
            self.features = X_train
            self.labels = y_train
        else:
            self.features = X_val
            self.labels = y_val
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {'x': feature, 'label': label}
    