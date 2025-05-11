import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time

from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

from datasets.datasets import get_dataset
from configs.config import load_train_config


if __name__ == "__main__":
    cfg = load_train_config("./邪王真眼/configs/previous_class07_svm.py")
    ngram_range = cfg.get('ngram_range')
    C = cfg.get('c')
    kernel = cfg.get('kernel')

    train_dataset, val_dataset = get_dataset(cfg)
    train_dataset = train_dataset.dataset
    val_dataset = val_dataset.dataset

    train_texts = []
    train_labels = []
    for batch_data in train_dataset:
        train_texts.append(batch_data.pop("x"))
        train_labels.append(batch_data.pop("label"))

    val_texts = []
    val_labels = []
    for batch_data in val_dataset:
        val_texts.append(batch_data.pop("x"))
        val_labels.append(batch_data.pop("label"))

    model = make_pipeline(
        TfidfVectorizer(ngram_range=ngram_range),
        SVC(probability=True, kernel=kernel, C=C)
    )

    model.fit(train_texts, train_labels)

    start_time = time.time()

    val_pred = model.predict(val_texts)

    total_time = time.time() - start_time
    print(f"验证耗时：{total_time}")

    accuracy = accuracy_score(val_labels, val_pred)
    print("Validation Accuracy:", accuracy)
