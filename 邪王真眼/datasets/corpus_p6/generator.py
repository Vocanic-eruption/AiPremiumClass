import os
import torch
import random

from typing import Any, Dict
from torch.utils.data import Dataset


class CorpusP6(Dataset):
    def __init__(self, _: str, cfg: Dict[str, Any]):
        self.window_length = cfg.get("window_length")
        self.sample_length = cfg.get("sample_length")

        data_root = cfg.get("data_root")
        corpus_path = os.path.join(data_root, 'corpus', '体育.txt')
        
        with open(corpus_path, encoding="utf8") as f:
            self.corpus = f.read()
    
        vocab = sorted(set(self.corpus))
        vocab.insert(0, "<UNK>")
        self.vocab = {ch: idx for idx, ch in enumerate(vocab)}
        
        cfg['vocab_size'] = len(vocab)
        cfg['num_classes'] = len(vocab)

        self.corpus_len = len(self.corpus)

    def __len__(self) -> int:
        return self.sample_length

    def __getitem__(self, _: int):
        start = random.randint(0, self.corpus_len - self.window_length - 1)
        end = start + self.window_length
        window = self.corpus[start:end]
        target = self.corpus[end]

        x = [self.vocab.get(ch, self.vocab["<UNK>"]) for ch in window]
        label = self.vocab.get(target, self.vocab["<UNK>"])

        x = torch.LongTensor(x)
        label = torch.LongTensor([label]).squeeze()

        return {'x': x, 'label': label}
