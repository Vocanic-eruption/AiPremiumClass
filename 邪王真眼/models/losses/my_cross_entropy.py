from torch import Tensor, zeros_like


def softmax(logits: Tensor) -> Tensor:
    max_logits = logits.amax(dim=1, keepdim=True)
    stabilized = logits - max_logits
    exp_logits = stabilized.exp()
    sum_exp = exp_logits.sum(dim=1, keepdim=True)
    probs = exp_logits / sum_exp
    return probs


def to_one_hot(logits: Tensor, target: Tensor) -> Tensor:
    one_hot = zeros_like(logits)
    one_hot.scatter_(1, target.unsqueeze(1), 1.0)
    return one_hot


def my_cross_entropy(input: Tensor, target: Tensor) -> Tensor:
    probs = softmax(input)
    one_hot = to_one_hot(input, target)
    log_probs = probs.log()
    loss = -(one_hot * log_probs).sum(dim=1)
    return loss.mean()