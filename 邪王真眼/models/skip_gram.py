import torch
import torch.nn as nn

from typing import Any, Dict


class SkipGram(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super(SkipGram, self).__init__()
        vocab_size = cfg["vocab_size"]
        embedding_dim = cfg["embedding_dim"]

        self.embedding_in = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_out = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        x = self.embedding_in(x)
        out = torch.matmul(x, self.embedding_out.weight.t())
        return {'out': out}
