from .plugins_base import PluginBase
from trainer.predicter import PluginType, PredictContext


class LogValResultPlugin(PluginBase):
    plugin_hooks = {
        PluginType.BATCH_END: "cal_acc",
        PluginType.VAL_END: "log"
    }

    def __init__(self):
        self.correct_data = 0
        self.total_data = 0

    def cal_acc(self, ctx: PredictContext):
        labels = ctx.labels
        outputs = ctx.outputs
        bs = labels.size(0)

        self.correct_data += (outputs.argmax(dim=1) == labels).sum().item()
        self.total_data += bs

    def log(self, ctx: PredictContext):
        acc = self.correct_data / self.total_data

        msg = {
            "mode":   "val",
            "loss":   round(ctx.avg_loss, 6),
            "acc":    round(acc, 6),
        }
        
        print(str(msg))
