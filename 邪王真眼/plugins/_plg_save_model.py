import os
import torch

from .plugins_base import PluginBase
from trainer.trainer import PluginType, TrainContext


class ModelSaverPlugin(PluginBase):
    plugin_hooks = {
        PluginType.TRAIN_END: "save"
    }

    def save(self, ctx: TrainContext):
        if not self.check_key(ctx.workspace, "results_folder", f"{type(self).__name__ }need variable: results_folder"):
            return
        checkpoint_path = ctx.workspace["results_folder"]
        os.makedirs(checkpoint_path, exist_ok=True)

        save_path = os.path.join(
            checkpoint_path,
            f"{ctx.cfg.get('model').value}.pth"
        )

        torch.save({
            "epoch": ctx.epoch,
            "model_state_dict": ctx.model.state_dict(),
            "optimizer_state_dict": ctx.optimizer.state_dict(),
            "scheduler_state_dict": ctx.workspace["scheduler"].state_dict() if self.check_key(ctx.workspace, "scheduler") else None,
        }, save_path)
