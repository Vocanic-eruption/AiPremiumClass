import torch
import plugins

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from torch.utils.data import DataLoader

from models.losses.loss_functions import get_loss_function
from models.optimizers.optimizers import get_optimizer
from models.models import get_model
from datasets.datasets import get_dataset


@dataclass
class TrainContext:
    cfg: Optional[Any] = None
    device: Optional[torch.device] = None
    model: Any = None
    train_loader: Optional[DataLoader] = None
    val_loader: Optional[DataLoader] = None
    optimizer: Any = None
    criterion: Any = None

    epoch: int = -1
    batch: int = -1
    running_loss: float = 0.0
    avg_loss: float = 0.0

    inputs: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    outputs: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None

    workspace: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    

class PluginType(Enum):
    TRAIN_BEGIN = 0
    TRAIN_END = 1

    EPOCH_BEGIN = 2
    EPOCH_END = 3

    BATCH_BEGIN = 4
    BATCH_END = 5


class Trainer(plugins.PluginsItem):
    def __init__(self, cfg):
        super().__init__(plugin_points=list(PluginType))

        self.cfg = cfg
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_loader, self.val_loader = get_dataset(cfg)

        self.model = get_model(cfg).to(self.device)

        self.optimizer = get_optimizer(cfg, self.model)

        self.criterion = get_loss_function(cfg)

        self.build_context(TrainContext)


    def train(self):
        self.run_plugins(PluginType.TRAIN_BEGIN, self.context)
        
        for epoch in range(self.cfg.get('epochs')):
            self.context.epoch = epoch
            self.run_plugins(PluginType.EPOCH_BEGIN, self.context)
            self.model.train()
            running_loss = 0.0

            for batch, batch_data in enumerate(self.train_loader):
                self.context.batch = batch
                batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
                label = batch_data.pop("label")

                self.context.inputs = batch_data
                self.context.labels = label

                self.run_plugins(PluginType.BATCH_BEGIN, self.context)

                self.optimizer.zero_grad()
                outputs = self.model(**batch_data)
                self.context.outputs = outputs
                if 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    outputs_t = {'input': outputs['out'], **{k: v for k, v in outputs.items() if k != 'out'}}
                    loss = self.criterion(target=self.context.labels, **outputs_t)
                
                self.context.loss = loss
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                self.context.running_loss = running_loss
                self.run_plugins(PluginType.BATCH_END, self.context)
                
            avg_loss = running_loss / (batch + 1)
            self.context.avg_loss = avg_loss
            self.run_plugins(PluginType.EPOCH_END, self.context)
            
        self.run_plugins(PluginType.TRAIN_END, self.context)
