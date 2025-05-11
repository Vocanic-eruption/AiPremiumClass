import re
import torch

from collections import defaultdict
from .plugins_base import PluginBase
from trainer.trainer import PluginType, TrainContext


class ValCRFEvaluationPlugin(PluginBase):
    plugin_hooks = {
        PluginType.EPOCH_END: "evaluate"
    }

    def __init__(self):
        self.history = []
    
    def evaluate(self, ctx: TrainContext):
        if (ctx.epoch + 1) % 1 != 0:
            ctx.workspace['val_acc'] = None
            return

        model, val_loader, device = (
            ctx.model, ctx.val_loader, ctx.device
        )

        if model is None or val_loader is None:
            raise RuntimeError("Validation plugin requires model and val_loader in context.")

        model.eval()
        total_loss = 0.0
        entities_correct = entities_total = 0

        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = {k: v.to(device) for k, v in batch_data.items()}
                labels = batch_data.pop("label")

                outputs = model(**batch_data)
                emissions, mask = outputs['out'], outputs['mask']
                pred_labels = model.decode(emissions, mask)

                correct, total = self.decode(labels, pred_labels, mask)
                entities_correct += correct
                entities_total += total

                loss = outputs['loss']
                total_loss += loss.item()
        
        acc = entities_correct / entities_total

        val_loss = total_loss / len(val_loader)


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

        model.train()

    def decode(self, labels, pred, mask):
        batch_size = labels.size(0)
        true_entities = []
        pred_entities = []

        labels = labels.cpu().numpy()
        mask = mask.cpu().numpy()

        for i in range(batch_size):
            true_seq = labels[i][mask[i] == 1]
            pred_seq = pred[i]

            true_entities += self.extract_entities(true_seq.tolist())
            pred_entities += self.extract_entities(pred_seq)

        correct = len([e for e in pred_entities if e in true_entities])

        return correct, len(true_entities)
    
    def extract_entities(self, seq):
        entities = []
        seq = ''.join(map(str, seq))

        patterns = {
            0: r"(04+)",  # LOCATION
            1: r"(15+)",  # ORGANIZATION
            2: r"(26+)",  # PERSON
            3: r"(37+)",  # TIME
        }

        for type_id, pattern in patterns.items():
            for match in re.finditer(pattern, seq):
                start = match.start()
                end = match.end() - 1
                entities.append( (start, end, type_id) )

        return entities
