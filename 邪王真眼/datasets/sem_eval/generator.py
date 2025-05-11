import os
import torch
import pandas as pd

from typing import Any, Dict
from torch.utils.data import Dataset
from transformers import BertTokenizer


class SemEvalDataset(Dataset):
    _FILES = {
        "train": "2018-E-c-En-train.txt",
        "val":   "2018-E-c-En-dev.txt",
        "test":  "2018-E-c-En-test.txt",
    }

    def __init__(self, split: str, cfg: Dict[str, Any]):
        if split not in self._FILES:
            raise ValueError(f"Unknown split '{split}', expected one of {list(self._FILES)}")
        
        base_dir = os.path.join(
            cfg["data_root"],
            "SemEval 2018 - task E-c",
            "archive"
        )
        data_file = os.path.join(base_dir, self._FILES[split])

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = cfg.get("max_len")

        df = pd.read_csv(data_file, delimiter="\t", encoding="utf-8")
        self.texts  = df["Tweet"].values
        self.labels = df.drop(columns=["ID", "Tweet"]).values
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        label = torch.tensor(label, dtype=torch.float32)

        return {'x': input_ids, 'label': label, 'mask': attention_mask}
