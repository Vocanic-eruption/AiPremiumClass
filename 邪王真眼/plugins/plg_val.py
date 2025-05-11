import torch

from .plugins_base import PluginBase
from trainer.trainer import PluginType, TrainContext
from models.losses.loss_functions import LossFunctionType


class ValEvaluationPlugin(PluginBase):
    plugin_hooks = {
        PluginType.EPOCH_END: "evaluate"
    }

    def __init__(self):
        self.history = []
    
    @staticmethod
    def _single_label_correct(logits, targets):
        preds = logits.argmax(dim=-1)
        return (preds == targets).sum().item()
    
    @staticmethod
    def _multi_label_correct(logits, labels):
        preds       = (torch.sigmoid(logits) > 0.5)
        targets     = (labels > 0.5)
        return preds.eq(targets).all(dim=1).sum().item()
    
    def evaluate(self, ctx: TrainContext):
        if (ctx.epoch + 1) % 1 != 0:
            ctx.workspace['val_acc'] = None
            ctx.workspace['val_loss'] = None
            return

        loss_function : LossFunctionType = ctx.cfg.get("loss_function")
        
        model, val_loader, criterion, device = (
            ctx.model, ctx.val_loader, ctx.criterion, ctx.device
        )

        if model is None or val_loader is None:
            raise RuntimeError("Validation plugin requires model and val_loader in context.")

        model.eval()
        total_loss = 0.0
        tot_correct = 0

        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = {k: v.to(device) for k, v in batch_data.items()}
                label = batch_data.pop("label")

                outputs = model(**batch_data)
                logits = outputs['out']
                if 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    outputs_t = {'input': outputs['out'], **{k: v for k, v in outputs.items() if k != 'out'}}
                    loss = criterion(target=label, **outputs_t)
                total_loss += loss.item()

                if loss_function == LossFunctionType.BCE_WITH_LOGITS:
                    tot_correct += self._multi_label_correct(logits, label)
                else:
                    tot_correct += self._single_label_correct(logits, label)
        

        n = len(val_loader.dataset)
        val_loss = total_loss / len(val_loader)
        acc = tot_correct / n


        self.history.append({
            "epoch": ctx.epoch,
            "val_loss": val_loss,
            "val_acc": acc,
        })

        msg = {
            "mode":   "val",
            "epoch":  ctx.epoch + 1,
            "lr":     round(ctx.optimizer.param_groups[0]['lr'], 6),
            "loss":   round(val_loss, 6),
            "acc":    round(acc, 6),
        }
        
        if self.check_key(ctx.workspace, "logger"):
            ctx.workspace["logger"](str(msg))

        ctx.workspace['val_acc'] = acc
        ctx.workspace['val_loss'] = val_loss

        model.train()
