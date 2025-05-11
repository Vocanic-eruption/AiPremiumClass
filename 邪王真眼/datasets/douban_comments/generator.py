import os
import torch
import pandas as pd

from typing import Any, Dict
from torch.utils.data import Dataset

from utils.tokenizer import Tokenizer


class DoubanCommentDataset(Dataset):
    def __init__(self, split: str, cfg: Dict[str, Any]):
        self.split = split
        self.csv_file_path = os.path.join(cfg.get("data_root"), 'DMSC.csv')
        self.split_ratio = cfg.get("split_ratio")
        self.data_max_num = cfg.get("data_max_num")
        self.seg_type = cfg.get("seg_type")

        self.data = self._load_data(cfg)
        
        if split == "train":
            cfg.update({
                'text_len': self.max_len,
                'vocab_size': len(self.vocab_to_index),
                '_vocab_to_index': self.vocab_to_index,
                '_index_to_vocab': self.index_to_vocab
            })

    def _load_data(self, cfg: Dict) -> pd.DataFrame:
        raw = pd.read_csv(self.csv_file_path, usecols=["Comment", "Star"])
        data = raw.iloc[:self.data_max_num].reset_index(drop=True)
        print(f"data use: {min(self.data_max_num, len(raw))} / {len(raw)}")

        split_idx = int(len(data) * self.split_ratio)
        data = data.iloc[:split_idx] if self.split == "train" else data.iloc[split_idx:]
        comments = data["Comment"].tolist()

        if self.split == "train":
            tokenizer = Tokenizer(comments, self.seg_type)
            self.max_len = tokenizer.max_len
            self.vocab_to_index = tokenizer.vocab_to_index
            self.index_to_vocab = tokenizer.index_to_vocab
        else:
            tokenizer = Tokenizer([], self.seg_type)
            self.vocab_to_index = tokenizer.vocab_to_index = cfg["_vocab_to_index"]
            self.index_to_vocab = tokenizer.index_to_vocab = cfg["_index_to_vocab"]
            self.max_len = cfg["text_len"]

        seqs = [tokenizer.encode(text, self.max_len) for text in comments]

        data["Comment"] = pd.Series(list(seqs), index=data.index)
        data.loc[:, "Star"] = (data["Star"] <= 2).astype(int)
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        content = torch.tensor(self.data.iloc[idx]["Comment"], dtype=torch.long)
        label   = torch.tensor(self.data.iloc[idx]["Star"],  dtype=torch.long)
        return {'x': content, 'label': label}
    