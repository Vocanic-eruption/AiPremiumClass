import os
import jieba
import pandas as pd

from typing import Any, Dict
from torch.utils.data import Dataset


class ECommerceComments(Dataset):
    def __init__(self, split: str, cfg: Dict[str, Any]):
        self.data_root = cfg.get("data_root")
        self.csv_file_path = os.path.join(self.data_root, 'E-commerce-Comments', 'comments.csv')
        self.data = pd.read_csv(self.csv_file_path)
        self.split = split

        grouped = self.data.groupby('label')
        subsets = []
        for label, group in grouped:
            group = group.sort_index()
            n = len(group)
            split_index = int(n * 0.7)
            if split == 'train':
                subset = group.iloc[:split_index]
            elif split == 'val':
                subset = group.iloc[split_index:]
            else:
                subset = group

            subsets.append(subset)
        self.data = pd.concat(subsets).reset_index(drop=True)
        
        self.texts = self.data['review'].tolist()
        self.labels = self.data['label'].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        return {'x': text, 'label': label}
    
    def generate_fasttext_file(self):
        output_dir = os.path.join(self.data_root, 'E-commerce-Comments')
        
        os.makedirs(output_dir, exist_ok=True)
        
        filename = os.path.join(output_dir, f"ECommerceComments_{self.split}.txt")
        
        if not os.path.exists(filename):
            with open(filename, 'w', encoding='utf-8') as f:
                for text, label in zip(self.texts, self.labels):
                    tokenized_text = ' '.join(jieba.cut(text))
                    f.write(f"__label__{label} {tokenized_text}\n")
        
        return filename
