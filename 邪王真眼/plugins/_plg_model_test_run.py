import torch

from .plugins_base import PluginBase
from trainer.trainer import PluginType, TrainContext


class PluginModelTestRun(PluginBase):
    plugin_hooks = {
        PluginType.TRAIN_BEGIN: "test_run"
    }

    def test_run(self, ctx: TrainContext):
        try:
            batch_data = next(iter(ctx.train_loader))
            device = ctx.device
            batch_data = {k: v.to(device) for k, v in batch_data.items()}
            _ = batch_data.pop("label")
            ctx.model.eval()

            with torch.no_grad():
                _ = ctx.model(**batch_data)['out']

        except Exception as e:
            raise RuntimeError(f"Model test run failed: {type(e).__name__}: {e}")
