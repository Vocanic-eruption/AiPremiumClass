import os
import torch
import plugins

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from torch.utils.data import DataLoader

from models.losses.loss_functions import get_loss_function
from models.models import get_model
from datasets.datasets import get_dataset


@dataclass
class PredictContext:
    cfg: Optional[Any] = None
    device: Optional[torch.device] = None
    model: Any = None
    val_loader: Optional[DataLoader] = None
    criterion: Any = None

    batch: int = -1
    running_loss: float = 0.0
    avg_loss: float = 0.0

    inputs: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    outputs: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None

    workspace: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    

class PluginType(Enum):
    VAL_BEGIN = 0
    VAL_END = 1

    BATCH_BEGIN = 2
    BATCH_END = 3


class Predicter(plugins.PluginsItem):
    def __init__(self, cfg, model_path):
        super().__init__(plugin_points=list(PluginType))

        self.cfg = cfg
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        _, self.val_loader = get_dataset(cfg)

        self.model = get_model(cfg).to(self.device)

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.criterion = get_loss_function(cfg)

        self.build_context(PredictContext)

    def val(self):
        self.run_plugins(PluginType.VAL_BEGIN, self.context)
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch, batch_data in enumerate(self.val_loader):
                self.context.batch = batch
                batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
                label = batch_data.pop("label")

                self.context.inputs = batch_data
                self.context.labels = label

                self.run_plugins(PluginType.BATCH_BEGIN, self.context)

                outputs = self.model(**batch_data)
                self.context.outputs = outputs
                if 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    outputs_t = {'input': outputs['out'], **{k: v for k, v in outputs.items() if k != 'out'}}
                    loss = self.criterion(target=label, **outputs_t)
                
                self.context.loss = loss

                running_loss += loss.item()
                self.context.running_loss = running_loss
                self.run_plugins(PluginType.BATCH_END, self.context)
        
        avg_loss = running_loss / len(self.val_loader)
        self.context.avg_loss = avg_loss
        self.run_plugins(PluginType.VAL_END, self.context)
