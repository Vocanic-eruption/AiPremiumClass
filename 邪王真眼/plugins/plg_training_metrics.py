import torch

from .plugins_base import PluginBase
from trainer.trainer import PluginType, TrainContext
from models.losses.loss_functions import LossFunctionType


class TrainingMetricsPlugin(PluginBase):
    plugin_hooks = {
        PluginType.BATCH_END: "cal_acc",
        PluginType.EPOCH_END: "log_metrics",
    }

    def __init__(self):
        self.train_loss = []
        self.train_acc = []

        self.correct_data = 0
        self.total_data = 0
    
    @staticmethod
    def _single_label_correct(logits, targets):
        preds = logits.argmax(dim=-1)
        return (preds == targets).sum().item()
    
    @staticmethod
    def _multi_label_correct(logits, labels):
        preds       = (torch.sigmoid(logits) > 0.5)
        targets     = (labels > 0.5)
        return preds.eq(targets).all(dim=1).sum().item()
    
    def cal_acc(self, ctx: TrainContext):
        labels = ctx.labels
        outputs = ctx.outputs['out']
        loss_function : LossFunctionType = ctx.cfg.get("loss_function")
        
        if loss_function == LossFunctionType.BCE_WITH_LOGITS:
            correct = self._multi_label_correct(outputs, labels)
            total   = labels.size(0)
        else:
            correct = self._single_label_correct(outputs, labels)
            total   = labels.size(0)

        self.correct_data += correct
        self.total_data += total

        print(f'\rBatch: {ctx.batch+1}/{len(ctx.train_loader)}', end='', flush=True)
        if ctx.batch + 1 == len(ctx.train_loader):
            print('\r', end='', flush=True)

    def log_metrics(self, ctx: TrainContext):
        avg_loss = ctx.avg_loss

        self.train_loss.append(avg_loss)

        acc = self.correct_data / self.total_data
        self.correct_data = self.total_data = 0
        
        msg = {
            "mode": "train",
            "epoch": ctx.epoch + 1,
            "lr": round(ctx.optimizer.param_groups[0]['lr'], 6),
            "loss": round(avg_loss, 6),
            "acc": round(acc, 6),
        }

        ctx.workspace['train_acc'] = acc
        
        if self.check_key(ctx.workspace, "logger"):
            ctx.workspace["logger"](str(msg))
