import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch

from models.models import get_model
from datasets.datasets import get_dataset
from utils.cal_perplexity import cal_perplexity


if __name__ == "__main__":
    cfg_path = "./邪王真眼/previous_class06/result/config.json"
    with open(cfg_path, 'r', encoding='utf8') as f:
        cfg = json.load(f)
    
    dataset, _ = get_dataset(cfg)
    model = get_model(cfg).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    vocab = dataset.dataset.vocab
    window_length = cfg.get('window_length')

    sentence = "在欧巡赛扭转了自己此前不利的状态"
    
    print(cal_perplexity(sentence, model, vocab, window_length))