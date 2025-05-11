import json
import os
import torch

from typing import Any, Dict
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class NER(Dataset):
    def __init__(self, split: str, cfg: Dict[str, Any]):
        data_root = cfg.get("data_root")
        self.max_length = cfg.get("max_length")

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        with open(os.path.join(data_root, 'ner', 'schema.json'), encoding="utf-8") as f:
            self.schema = json.load(f)

        self.label2id = {label: idx for idx, label in enumerate(self.schema)}
        
        cfg["vocab_size"] = self.tokenizer.vocab_size
        cfg["num_classes"] = len(self.schema)
        
        split = 'test' if split == 'val' else 'train'
        self.data = self._load_data(os.path.join(data_root, 'ner', split))

    def _load_data(self, data_path: str) -> list:
        samples = []
        with open(data_path, encoding="utf-8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                if not segment.strip():
                    continue
                    
                tokens, labels = [], []
                for line in segment.split("\n"):
                    if not line.strip():
                        continue
                    char, label = line.split()
                    tokens.append(char)
                    labels.append(self.label2id[label])
                
                encoding = self.tokenizer(
                    tokens,
                    is_split_into_words=True,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                    return_attention_mask=True,
                    add_special_tokens=False
                )
                
                word_ids = encoding.word_ids()
                aligned_labels = self._align_labels_with_tokens(labels, word_ids)
                aligned_labels = aligned_labels[:self.max_length]
                
                samples.append({
                    "x": encoding["input_ids"].squeeze(0),
                    "mask": encoding["attention_mask"].squeeze(0),
                    "label": torch.LongTensor(aligned_labels),
                    "tag": torch.LongTensor(aligned_labels),
                })

        return samples

    def _align_labels_with_tokens(self, labels: list, word_ids: list) -> list:
        aligned_labels = []
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-1) # 使用crf这里必须是-1啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊赣
            else:
                aligned_labels.append(labels[word_idx])
        return aligned_labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]
