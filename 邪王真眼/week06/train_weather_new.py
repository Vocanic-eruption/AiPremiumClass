import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import plugins as plugins

from configs.config import load_train_config
from trainer.trainer import Trainer
from plg_change_label_shape import ChangeLabelShapePlugin
from plg_training_metrics_mse import TrainingMetricsMSEPlugin
from plg_val_mse import ValEvaluationMSEPlugin


if __name__ == "__main__":
    cfg = load_train_config("./邪王真眼/configs/week06_weather.py")
    trainer = Trainer(cfg)
    trainer.add_plugins([
        plugins.PluginModelTestRun,
        plugins.PluginInitInfo,
        plugins.PluginLogger,
        plugins.PluginSaveConfig,
        plugins.ModelSaverPlugin,
        TrainingMetricsMSEPlugin,
        ValEvaluationMSEPlugin,
        plugins.TensorBoardPlugin,
        ChangeLabelShapePlugin,
    ])
    trainer.train()

# tensorboard --logdir=./邪王真眼/week06/result

# 训练流程发生变化，一个batch需要多次循环调用model，融入不到这个框架，需要新的trainer
# 新脚本目前只能预测一天，原来的train_weather.py可以预测多天
# 这三个插件有点屎了