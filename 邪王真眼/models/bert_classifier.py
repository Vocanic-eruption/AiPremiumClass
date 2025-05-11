import torch.nn as nn

from torchcrf import CRF
from typing import Any, Dict
from transformers import BertModel, BertConfig


class BertClassifier(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super(BertClassifier, self).__init__()
        num_classes = cfg.get("num_classes")
        num_layers = cfg.get("num_layers")
        dropout = cfg.get("dropout")

        bert_cfg = BertConfig.from_pretrained(
            "bert-base-chinese",
            num_hidden_layers=num_layers,
            hidden_dropout_prob=dropout,
            output_hidden_states=False,
            return_dict=True
        )
        self.bert = BertModel.from_pretrained("bert-base-chinese", config=bert_cfg)
        self.classify = nn.Linear(768, num_classes)
        self.crf = CRF(num_tags=num_classes, batch_first=True)

    def decode(self, emissions, mask):
        return self.crf.decode(emissions, mask=mask.bool())
    
    def forward(self, x, mask, tag):
        outputs = self.bert(input_ids=x, attention_mask=mask)
        x = outputs.last_hidden_state
        emissions = self.classify(x)
        crf_mask = mask.bool()
        loss = -self.crf(emissions, tag, mask=crf_mask, reduction="mean")
        return {'out': emissions, 'mask': mask, 'loss': loss}