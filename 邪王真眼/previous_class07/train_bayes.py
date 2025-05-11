import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time

from models.bayes import BayesApproach
from datasets.datasets import get_dataset
from configs.config import load_train_config


if __name__ == "__main__":
    cfg = load_train_config("./邪王真眼/configs/previous_class07_bayes.py")
    train_dataset, val_dataset = get_dataset(cfg)
    train_dataset = train_dataset.dataset
    val_dataset = val_dataset.dataset
    model = BayesApproach(train_dataset)

    correct = 0
    total = len(val_dataset)

    start_time = time.time()

    for batch, batch_data in enumerate(val_dataset):
        label = batch_data.pop("label")
        pred_probs = model(**batch_data)['out']
        
        pred_label = max(pred_probs, key=pred_probs.get)
        if pred_label == label:
            correct += 1

    total_time = time.time() - start_time
    print(f"验证耗时：{total_time}")

    accuracy = correct / total
    print("Validation Accuracy:", accuracy)