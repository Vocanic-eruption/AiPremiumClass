import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import fasttext

from datasets.datasets import get_dataset
from configs.config import load_train_config


if __name__ == "__main__":
    cfg = load_train_config("./邪王真眼/configs/previous_class07_fasttext.py")
    dim = cfg.get('dim')
    epoch = cfg.get('epoch')
    lr = cfg.get('lr')

    train_dataset, val_dataset = get_dataset(cfg)
    train_dataset = train_dataset.dataset
    val_dataset = val_dataset.dataset

    train_file_path = train_dataset.generate_fasttext_file()
    val_file_path = val_dataset.generate_fasttext_file()

    model = fasttext.train_supervised(input=train_file_path, dim=dim, epoch=epoch, lr=lr) # 报错就是中文路径

    start_time = time.time()

    result = model.test(val_file_path)

    total_time = time.time() - start_time
    print(f"验证耗时：{total_time}")

    print(f"验证集准确率: {result[1]*100:.2f}%")
