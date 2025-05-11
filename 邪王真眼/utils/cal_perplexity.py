import torch
import torch.nn as nn
import math

from typing import Dict


def cal_perplexity(
    text: str,
    model: nn.Module,
    vocab: Dict[str, int],
    window_length: int
) -> float:
    """
    计算给定自回归语言模型对一段文本的困惑度（perplexity）。

    对长度为 N 的序列 x₁…xₙ，困惑度定义为
        PP = exp ( - (1/N) * Σ_{i=1}^N log p(xᵢ | x₍<ᵢ₎) ).

    :param text: 待评估文本
    :param model: 语言模型，forward(x) 返回 logits 或未归一化的分布
    :param vocab: 字符到索引的映射，必须包含 "<UNK>"
    :param window_length: 上下文窗口大小
    :return: 困惑度（float）
    """
    model.eval()
    device = next(model.parameters()).device

    unk_idx = vocab["<UNK>"]
    total_log_prob = 0.0
    N = len(text)

    with torch.no_grad():
        for i in range(1, N):
            start = max(0, i - window_length)
            window = text[start:i]

            idxs = [vocab.get(ch, unk_idx) for ch in window]
            x = torch.tensor([idxs], dtype=torch.long, device=device)

            logits = model(x)['out']
            log_probs = torch.log_softmax(logits, dim=-1)

            target_idx = vocab.get(text[i], unk_idx)
            total_log_prob += log_probs[0, target_idx].item()

    neg_avg_log_prob = - total_log_prob / N
    perplexity = math.exp(neg_avg_log_prob)
    return perplexity
