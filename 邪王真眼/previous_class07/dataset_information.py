import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from datasets.datasets import get_dataset
from configs.config import load_train_config


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


if __name__ == "__main__":
    cfg = load_train_config("./邪王真眼/configs/previous_class07_bayes.py")
    train_dataset, val_dataset = get_dataset(cfg)
    train_dataset = train_dataset.dataset
    val_dataset = val_dataset.dataset

    print(f'数据集长度: {len(train_dataset)}')
    label_counts = train_dataset.data['label'].value_counts()
    print(f"正样本数量: {label_counts.get(1, 0)}")
    print(f"负样本数量: {label_counts.get(0, 0)}")

    lengths = []
    for batch, batch_data in enumerate(val_dataset):
        text = batch_data.pop("x")
        lengths.append(len(text))

    log_lengths = np.log1p(lengths)

    avg_length = np.mean(lengths)
    median_length = np.median(lengths)

    avg_log_length = np.log1p(avg_length)
    median_log_length = np.log1p(median_length)

    plt.figure(figsize=(10, 6))

    n, bins, patches = plt.hist(log_lengths, bins=30, alpha=0.7, edgecolor='black')

    plt.axvline(avg_log_length, color='r', linestyle='dashed', linewidth=2,
                label=f'平均长度: {avg_length:.2f}')
    plt.axvline(median_log_length, color='g', linestyle='dashed', linewidth=2,
                label=f'中位数: {median_length:.2f}')

    plt.title('文本长度分布')
    plt.xlabel('文本长度（字符数）')
    plt.ylabel('样本数量')

    def log1p_to_normal_formatter(x, _):
        return f"{np.expm1(x):.1f}"

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(log1p_to_normal_formatter))

    plt.legend()
    plt.show()
