import os

from torch.utils.tensorboard import SummaryWriter

from .plugins_base import PluginBase
from trainer.trainer import PluginType, TrainContext


class TensorBoardPlugin(PluginBase):
    plugin_hooks = {
        PluginType.EPOCH_END: "write"
    }

    def __init__(self):
        self.writer = None
    
    def write(self, ctx: TrainContext):
        if self.writer is None:
            self.writer = SummaryWriter(os.path.join(ctx.cfg.get('result_path'), "runs"))
        
        model_type = ctx.cfg.get('model')
        uuid = ctx.workspace.get('uuid') if self.check_key(ctx.workspace, "uuid") else '0'
        self.writer.add_scalar(f'{model_type}_Loss_{uuid}/train', ctx.avg_loss, ctx.epoch)

        if self.check_key(ctx.workspace, "train_acc"):
            train_acc = ctx.workspace.get('train_acc')
            self.writer.add_scalar(f'{model_type}_Accuracy_{uuid}/train', train_acc, ctx.epoch)
        
        if self.check_key(ctx.workspace, "train_mae"):
            train_mae = ctx.workspace.get('train_mae')
            self.writer.add_scalar(f'{model_type}_Mae_{uuid}/train', train_mae, ctx.epoch)
        
        if self.check_key(ctx.workspace, "val_loss"):
            val_loss = ctx.workspace.get('val_loss')
            self.writer.add_scalar(f'{model_type}_Loss_{uuid}/val', val_loss, ctx.epoch)
        
        if self.check_key(ctx.workspace, "val_acc"):
            val_acc = ctx.workspace.get('val_acc')
            self.writer.add_scalar(f'{model_type}_Accuracy_{uuid}/val', val_acc, ctx.epoch)

        if self.check_key(ctx.workspace, "val_mae"):
            val_mae = ctx.workspace.get('val_mae')
            self.writer.add_scalar(f'{model_type}_Mae_{uuid}/val', val_mae, ctx.epoch)