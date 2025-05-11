import os
import pandas as pd
import torch
from typing import Any, Dict
from torch.utils.data import Dataset

from utils.word2vec_vectorizer import BERTVectorizer


class ECommerceComments(Dataset):
    def __init__(self, split: str, cfg: Dict[str, Any]):
        super().__init__()
        self.split = split

        self.tokenizer = BERTVectorizer()

        data_root = cfg.get('data_root')
        self.max_length = cfg.get('max_length')
        csv_path = os.path.join(data_root, 'E-commerce-Comments', 'comments.csv')
        raw = pd.read_csv(csv_path)

        grouped = raw.groupby('label')
        parts = []
        for label, group in grouped:
            group = group.sort_index()
            n = len(group)
            split_idx = int(n * 0.7)
            if split == 'train':
                part = group.iloc[:split_idx]
            elif split == 'val':
                part = group.iloc[split_idx:]
            else:
                part = group
            parts.append(part)

        data = pd.concat(parts).reset_index(drop=True)
        self.texts = data['review'].tolist()
        self.labels = data['label'].tolist()

        cfg['vocab_size'] = self.tokenizer.vocab_size
        self.num_classes = max(self.labels) + 1

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        input_ids, attention_mask = self.tokenizer.text_to_indices(
            text,
            max_length=self.max_length
        )

        label_tensor = torch.zeros(self.num_classes, dtype=torch.float)
        label_tensor[label] = 1.0

        return {'x': input_ids, 'mask': attention_mask, 'label': label_tensor}
