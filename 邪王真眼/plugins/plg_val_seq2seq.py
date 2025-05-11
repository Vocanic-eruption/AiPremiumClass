import torch

from .plugins_base import PluginBase
from trainer.trainer import PluginType, TrainContext


class ValEvaluationSeq2SeqPlugin(PluginBase):
    plugin_hooks = {
        PluginType.EPOCH_END: "evaluate"
    }

    def __init__(self):
        self.history = []
    
    @staticmethod
    def _single_label_correct(logits, targets):
        preds = logits.argmax(dim=-2)
        return (preds == targets).sum().item()
    
    def evaluate(self, ctx: TrainContext):
        if (ctx.epoch + 1) % 1 != 0:
            ctx.workspace['val_acc'] = None
            ctx.workspace['val_loss'] = None
            return

        model, val_loader, criterion, device = (
            ctx.model, ctx.val_loader, ctx.criterion, ctx.device
        )

        if model is None or val_loader is None:
            raise RuntimeError("Validation plugin requires model and val_loader in context.")

        model.eval()
        total_loss = 0.0
        tot_correct = 0
        total_valid_tokens = 0
        pad_token_id = val_loader.dataset.vocab["<PAD>"] 

        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = {k: v.to(device) for k, v in batch_data.items()}
                label = batch_data.pop("label")
                
                mask = (label != pad_token_id)
                valid_tokens = mask.sum().item()
                
                outputs = model(**batch_data)
                logits = outputs['out']
                
                if 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    outputs_t = {
                        'input': logits,
                        'target': label
                    }
                    loss = criterion(**outputs_t)
                
                total_loss += loss.item()

                if logits.dim() == 3:
                    preds = logits.argmax(dim=1)
                else:
                    preds = logits.argmax(dim=-1)

                correct = (preds == label) & mask
                tot_correct += correct.sum().item()
                total_valid_tokens += valid_tokens
        
        val_loss = total_loss / len(val_loader)
        acc = tot_correct / total_valid_tokens if total_valid_tokens > 0 else 0

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
