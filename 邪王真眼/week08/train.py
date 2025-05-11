import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import plugins as plugins

from configs.config import load_train_config
from trainer.trainer import Trainer


if __name__ == "__main__":
    cfg = load_train_config("./邪王真眼/configs/week08.py")
    trainer = Trainer(cfg)
    trainer.add_plugins([
        plugins.PluginModelTestRun,
        plugins.PluginInitInfo,
        plugins.PluginLogger,
        plugins.PluginSaveConfig,
        plugins.ModelSaverPlugin,
        plugins.TrainingMetricsSeq2SeqPlugin,
        plugins.ValEvaluationSeq2SeqPlugin,
        plugins.TensorBoardPlugin,
        plugins.PluginScheduler,
    ])
    trainer.train()
