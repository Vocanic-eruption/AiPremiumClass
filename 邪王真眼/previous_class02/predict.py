import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import plugins as plugins

from configs.config import load_train_config
from trainer.predicter import Predicter


if __name__ == "__main__":
    cfg = load_train_config("./邪王真眼/configs/previous_class02.py")
    predicter = Predicter(cfg, "./邪王真眼/previous_class02/result/resnet.pth")
    predicter.add_plugins([
        plugins.LogValResultPlugin,
    ])
    predicter.val()
