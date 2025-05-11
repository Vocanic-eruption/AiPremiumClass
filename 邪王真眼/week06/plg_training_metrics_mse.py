import torch.nn as nn

from plugins.plugins_base import PluginBase
from trainer.trainer import PluginType, TrainContext


class TrainingMetricsMSEPlugin(PluginBase):
    plugin_hooks = {
        PluginType.BATCH_END: "cal_mae",
        PluginType.EPOCH_END: "log_metrics",
    }

    def __init__(self):
        self.train_loss = []
        self.train_mae = []

        self.running_mae = 0
        self.mae_loss = nn.L1Loss()
    
    def cal_mae(self, ctx: TrainContext):
        labels = ctx.labels
        outputs = ctx.outputs['out']

        mae = self.mae_loss(outputs, labels)
        self.running_mae += mae.item()

        print(f'\rBatch: {ctx.batch+1}/{len(ctx.train_loader)}', end='', flush=True)
        if ctx.batch + 1 == len(ctx.train_loader):
            print('\r', end='', flush=True)

    def log_metrics(self, ctx: TrainContext):
        avg_loss = ctx.avg_loss
        mae = self.running_mae / len(ctx.train_loader)

        self.train_loss.append(avg_loss)
        self.train_mae.append(mae)
        
        self.running_mae = 0
        
        msg = {
            "mode": "train",
            "epoch": ctx.epoch + 1,
            "lr": round(ctx.optimizer.param_groups[0]['lr'], 6),
            "loss": round(avg_loss, 6),
            "mae": round(mae, 6),
        }

        ctx.workspace['train_mae'] = mae
        
        if self.check_key(ctx.workspace, "logger"):
            ctx.workspace["logger"](str(msg))
