import os
import torch

from typing import Any, Dict
from torch.utils.data import Dataset

from utils.jieba_segmenter import JiebaSegmenter


class Skipgram(Dataset):
    def __init__(self, split: str, cfg: Dict[str, Any]):
        if split not in ('train', 'val'):
            raise ValueError(f"Unknown split: {split}. Use 'train' or 'val'.")

        self.corpus_file = os.path.join(cfg.get("data_root"), 'corpus.txt')
        self.window_length = cfg.get("window_length")
        self.use_lines = cfg.get("use_lines")
        self.split = split

        total = self.use_lines
        split_line = int(total * 0.7)

        self.sentences = []
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= total:
                    break
                if self.split == 'train' and idx >= split_line:
                    continue
                if self.split == 'val'   and idx <  split_line:
                    continue
                text = line.strip()
                if text:
                    self.sentences.append(JiebaSegmenter.cut(text))
                    print(f"\rload lines [{idx+1}/{total}]", end='', flush=True)

        if self.split == 'train':
            self.word_counts = {}
            for sentence in self.sentences:
                for word in sentence:
                    self.word_counts[word] = self.word_counts.get(word, 0) + 1
            self.vocab       = list(self.word_counts.keys())
            self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
            self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}

            self.unk_idx    = len(self.vocab)
            self.vocab_size = len(self.vocab) + 1

            cfg['_word_to_idx'] = self.word_to_idx
            cfg['_idx_to_word'] = self.idx_to_word
            cfg['_unk_idx']     = self.unk_idx
            cfg['vocab_size']  = self.vocab_size

        else:
            self.word_to_idx = cfg['_word_to_idx']
            self.idx_to_word = cfg['_idx_to_word']
            self.unk_idx     = cfg['_unk_idx']
            self.vocab_size  = cfg['vocab_size']

        self.pairs = []
        for sentence in self.sentences:
            indices = [self.word_to_idx.get(w, self.unk_idx) for w in sentence]
            for center_pos, center_id in enumerate(indices):
                for offset in range(-self.window_length, self.window_length + 1):
                    ctx_pos = center_pos + offset
                    if offset == 0 or ctx_pos < 0 or ctx_pos >= len(indices):
                        continue
                    self.pairs.append((center_id, indices[ctx_pos]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return {
            'x': torch.tensor(center, dtype=torch.long),
            'label': torch.tensor(context, dtype=torch.long)
        }