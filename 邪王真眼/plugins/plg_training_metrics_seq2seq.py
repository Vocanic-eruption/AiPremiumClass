from .plugins_base import PluginBase
from trainer.trainer import PluginType, TrainContext


class TrainingMetricsSeq2SeqPlugin(PluginBase):
    plugin_hooks = {
        PluginType.BATCH_END: "cal_acc",
        PluginType.EPOCH_END: "log_metrics",
    }

    def __init__(self):
        self.train_loss = []
        self.train_acc = []

        self.correct_tokens = 0
        self.valid_tokens = 0
    
    def cal_acc(self, ctx: TrainContext):
        label = ctx.labels
        outputs = ctx.outputs['out']
        pad_token_id = ctx.train_loader.dataset.vocab["<PAD>"]

        mask = (label != pad_token_id)
        
        if outputs.dim() == 3:
            preds = outputs.argmax(dim=1)
        else:
            preds = outputs.argmax(dim=-1)

        correct = (preds == label) & mask
        self.correct_tokens += correct.sum().item()
        self.valid_tokens += mask.sum().item()

        print(f'\rBatch: {ctx.batch+1}/{len(ctx.train_loader)}', end='', flush=True)
        if ctx.batch + 1 == len(ctx.train_loader):
            print('\r', end='', flush=True)

    def log_metrics(self, ctx: TrainContext):
        avg_loss = ctx.avg_loss
        self.train_loss.append(avg_loss)

        acc = self.correct_tokens / self.valid_tokens if self.valid_tokens > 0 else 0
        self.train_acc.append(acc)
        
        self.correct_tokens = self.valid_tokens = 0
        
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
