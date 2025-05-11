import os
import torch

from typing import Any, Dict
from torch.utils.data import Dataset
from utils.jieba_segmenter import JiebaSegmenter


class CBOW(Dataset):
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
                if self.split == 'val' and idx < split_line:
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
            self.vocab        = list(self.word_counts.keys())
            self.word_to_idx  = {word: i for i, word in enumerate(self.vocab)}
            self.idx_to_word  = {i: word for word, i in self.word_to_idx.items()}
            
            self.unk_idx      = len(self.vocab)
            self.vocab_size   = len(self.vocab) + 1

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
            if len(indices) < 2 * self.window_length + 1:
                continue

            for center_pos in range(self.window_length, len(indices) - self.window_length):
                center_id = indices[center_pos]
                context_ids = []
                for offset in range(-self.window_length, self.window_length + 1):
                    if offset == 0:
                        continue
                    context_ids.append(indices[center_pos + offset])
                self.pairs.append((context_ids, center_id))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        context_ids, center_id = self.pairs[idx]
        return {
            'x': torch.tensor(context_ids, dtype=torch.long),
            'label': torch.tensor(center_id, dtype=torch.long)
        }
