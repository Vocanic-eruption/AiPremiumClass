import torch
import torch.nn as nn

from plugins.plugins_base import PluginBase
from trainer.trainer import PluginType, TrainContext


class ValEvaluationMSEPlugin(PluginBase):
    plugin_hooks = {
        PluginType.EPOCH_END: "evaluate"
    }

    def __init__(self):
        self.history = []
    
    def evaluate(self, ctx: TrainContext):
        if (ctx.epoch + 1) % 1 != 0:
            ctx.workspace['val_mae'] = None
            ctx.workspace['val_loss'] = None
            return

        model, val_loader, criterion, device = (
            ctx.model, ctx.val_loader, ctx.criterion, ctx.device
        )

        if model is None or val_loader is None:
            raise RuntimeError("Validation plugin requires model and val_loader in context.")

        model.eval()
        total_loss = 0.0
        running_mae = 0.0

        mae_loss = nn.L1Loss()
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = {k: v.to(device) for k, v in batch_data.items()}
                labels = batch_data.pop("label").squeeze(1)

                outputs = model(**batch_data)
                outputs_t = {'input': outputs['out'], **{k: v for k, v in outputs.items() if k != 'out'}}
                loss = criterion(target=labels, **outputs_t)
                
                mae = mae_loss(target=labels, **outputs_t)
                running_mae += mae.item()

                total_loss += loss.item()

        val_loss = total_loss / len(val_loader)
        mae = running_mae / len(val_loader)

        self.history.append({
            "epoch": ctx.epoch,
            "val_loss": val_loss,
            "val_mae": mae,
        })

        msg = {
            "mode":   "val",
            "epoch":  ctx.epoch + 1,
            "lr":     round(ctx.optimizer.param_groups[0]['lr'], 6),
            "loss":   round(val_loss, 6),
            "mae":    round(mae, 6),
        }
        
        if self.check_key(ctx.workspace, "logger"):
            ctx.workspace["logger"](str(msg))

        ctx.workspace['val_mae'] = mae
        ctx.workspace['val_loss'] = val_loss

        model.train()
