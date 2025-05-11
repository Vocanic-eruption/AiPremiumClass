from .plugins_base import PluginBase
from torch.optim.lr_scheduler import LambdaLR
from trainer.trainer import PluginType, TrainContext


class PluginScheduler(PluginBase):
    def __init__(self, mode='iter'):
        assert mode in ['epoch', 'iter'], "mode must be one of 'epoch', 'iter'"
        self.mode = mode
        self.plugin_hooks = {
            PluginType.TRAIN_BEGIN: "initialize_scheduler",
            PluginType.BATCH_END if mode == 'iter' else PluginType.EPOCH_END: "step"
        }

    def initialize_scheduler(self, ctx: TrainContext):
        if self.mode == 'iter':
            ctx.workspace["iter"] = 0
            total = ctx.cfg.get("epochs") * len(ctx.train_loader)
        else:
            total = ctx.cfg.get("epochs")

        lr_lambda = lambda x: (1 - x / total) ** 0.9
        optimizer = ctx.optimizer

        self.scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        ctx.workspace["scheduler"] = self.scheduler

    def step(self, ctx: TrainContext):
        if self.mode == 'iter':
            ctx.workspace["iter"] += 1
        self.scheduler.step()
