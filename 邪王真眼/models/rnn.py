import torch.nn as nn

from typing import Any, Dict
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super(RNN, self).__init__()
        rnn_type  = cfg.get("rnn_type").lower()
        input_size  = cfg.get("input_size")
        hidden_size = cfg.get("hidden_size", input_size)
        num_classes = cfg.get("num_classes")
        num_layers  = cfg.get("num_layers")
        dropout     = cfg.get("dropout", 0.0)
        vocab_size  = cfg.get("vocab_size", None)
        
        if vocab_size is not None:
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=input_size,
                padding_idx=1
            )
            
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers,
                             batch_first=True, bidirectional=False,
                             dropout=dropout)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                              batch_first=True, bidirectional=False,
                              dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                             batch_first=True, bidirectional=False,
                             dropout=dropout)
        elif rnn_type == 'birnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers,
                             batch_first=True, bidirectional=True,
                             dropout=dropout)
        else:
            raise ValueError(f"Unknown model type: {rnn_type}. "
                           f"Available options are: 'rnn', 'lstm', 'gru', 'birnn'")
        
        fc_input_size = hidden_size * (2 if rnn_type == 'birnn' else 1)

        self.fc = nn.Linear(fc_input_size, num_classes)
        
    def forward(self, x, mask=None):
        if hasattr(self, 'embedding'):
            x = self.embedding(x)
        
        if mask is not None:
            lengths = mask.sum(dim=1).long()
            
            packed = pack_padded_sequence(
                x, lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            packed_out, _ = self.rnn(packed)
            out_padded, _ = pad_packed_sequence(packed_out, batch_first=True)
            
            idx = (lengths - 1) \
                  .view(-1, 1, 1) \
                  .expand(-1, 1, out_padded.size(2))  # [B,1,H]
            out = out_padded.gather(1, idx).squeeze(1)  # [B, H]
        else:
            out, _ = self.rnn(x)
            out = out[:, -1, :]
        
        out = self.fc(out)
        return {'out': out}
