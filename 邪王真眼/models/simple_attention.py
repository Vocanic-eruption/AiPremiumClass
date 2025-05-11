import math
import torch
import torch.nn as nn

from enum import Enum, auto
from typing import Any, Dict


class MergeMode(Enum):
    CONCAT = auto()
    SUM = auto()
    MULTIPLY = auto()
    

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, merge_mode=MergeMode.CONCAT):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.merge_mode = merge_mode
        if not isinstance(merge_mode, MergeMode):
            raise ValueError(f"merge_mode必须是MergeMode枚举值当前为{type(merge_mode)}")

    def forward(self, token_seq):
        embedded = self.embedding(token_seq)
        outputs, hidden = self.rnn(embedded)

        forward_hidden = hidden[0]
        backward_hidden = hidden[1]
        
        if self.merge_mode == MergeMode.CONCAT:
            hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        elif self.merge_mode == MergeMode.SUM:
            sum_hidden = (forward_hidden + backward_hidden)
            hidden = torch.cat((sum_hidden, sum_hidden), dim=1)
        elif self.merge_mode == MergeMode.MULTIPLY:
            mul_hidden = (forward_hidden * backward_hidden)
            hidden = torch.cat((mul_hidden, mul_hidden), dim=1)
        
        return outputs, hidden



class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)

        self.scale_factor = math.sqrt(hidden_dim)

    def forward(self, enc_output, dec_output):
        Q = self.W_Q(dec_output)
        K = self.W_K(enc_output)
        V = self.W_V(enc_output)

        attn_scores = torch.bmm(Q, K.permute(0, 2, 1))
        attn_scores = attn_scores / self.scale_factor
        attn_weights = torch.softmax(attn_scores, dim=-1)
        c_t = torch.bmm(attn_weights, V)

        return c_t


class Decoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim * 2, batch_first=True)
        self.attention = Attention(hidden_dim * 2)
        self.attention_fc = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, input_dim)

    def forward(self, x, x_decode, init_hidden):
        x_decode = self.embedding(x_decode)
        x_decode, hidden = self.rnn(x_decode, init_hidden)

        c_t = self.attention(x, x_decode)
        cat_output = torch.cat((c_t, x_decode), dim=-1)
        out = torch.tanh(self.attention_fc(cat_output))

        logits = self.fc(out)
        return logits, hidden


class SimpleAttention(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        vocab_size  = cfg.get("vocab_size")
        embedding_dim  = cfg.get("embedding_dim")
        hidden_size  = cfg.get("hidden_size")
        merge_mode  = cfg.get("merge_mode", MergeMode.CONCAT)

        self.encoder = Encoder(vocab_size, embedding_dim, hidden_size, merge_mode)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_size)

    def forward(self, x, x_decode):
        x, encoder_hidden = self.encoder(x)
        encoder_hidden = encoder_hidden.unsqueeze(0)
        output, _ = self.decoder(x, x_decode, encoder_hidden)
        output = output.permute(0, 2, 1).contiguous()
        return {'out': output}
