import os
import torch

from typing import Any, Dict, Tuple
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class Couplet(Dataset):
    def __init__(self, split: str, cfg: Dict[str, Any]):
        self.split = "train" if split == "train" else "test"
        self.data_root = cfg["data_root"]
        self.max_len = cfg.get("max_len", 50)
        num_data = cfg.get("num_data", None)

        self.in_lines, self.out_lines = self._load_data()
        if num_data is not None:
            self.in_lines = self.in_lines[0:num_data]
            self.out_lines = self.out_lines[0:num_data]

        if split == "train":
            self.vocab = self._load_vocab()
            cfg.update({
                "vocab_size": len(self.vocab),
                "_vocab": self.vocab,
            })
        else:
            self.vocab = cfg.get("_vocab")

    def _load_data(self) -> Tuple[list, list]:
        split_dir = os.path.join(self.data_root, "couplet", self.split)
        in_path = os.path.join(split_dir, "in.txt")
        out_path = os.path.join(split_dir, "out.txt")

        with open(in_path, "r", encoding="utf-8") as f_in, \
             open(out_path, "r", encoding="utf-8") as f_out:
            in_lines = [line.strip().split() for line in f_in if line.strip()]
            out_lines = [line.strip().split() for line in f_out if line.strip()]

        assert len(in_lines) == len(out_lines), "输入输出对联数量不匹配"
        return in_lines, out_lines

    def _load_vocab(self) -> Tuple[dict, dict]:
        vocab_path = os.path.join(self.data_root, "couplet", "vocabs")
        
        vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        
        with open(vocab_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=4):
                token = line.strip()
                if token:
                    vocab[token] = idx
        
        return vocab

    def __len__(self) -> int:
        return len(self.in_lines)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        in_tokens = self.in_lines[idx]
        out_tokens = ["<BOS>"] + self.out_lines[idx] + ["<EOS>"]

        in_ids = [self.vocab.get(tk, 1) for tk in in_tokens]
        out_ids = [self.vocab.get(tk, 1) for tk in out_tokens]

        in_ids = in_ids[:self.max_len]
        out_ids = out_ids[:self.max_len]

        return {
            "x": torch.tensor(in_ids, dtype=torch.long),
            "x_decode": torch.tensor(out_ids[:-1], dtype=torch.long),
            "label": torch.tensor(out_ids[1:], dtype=torch.long)
        }

    def decode(self, token_ids: torch.Tensor) -> list:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy().tolist()
        
        id2token = {v: k for k, v in self.vocab.items()}
        
        tokens = []
        for tid in token_ids:
            token = id2token.get(tid, "<UNK>")
            if token in ["<BOS>", "<EOS>", "<PAD>"]:
                continue
            tokens.append(token)
        
        return "".join(tokens)

    @staticmethod
    def collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = pad_sequence(
            [item["x"] for item in batch],
            batch_first=True,
            padding_value=0
        )
        x_decode = pad_sequence(
            [item["x_decode"] for item in batch],
            batch_first=True,
            padding_value=0
        )
        label = pad_sequence(
            [item["label"] for item in batch],
            batch_first=True,
            padding_value=0
        )
        return {"x": x, "x_decode": x_decode, "label": label}