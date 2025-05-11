import time

from .plugins_base import PluginBase
from trainer.trainer import PluginType, TrainContext


class PluginTikTok(PluginBase):
    plugin_hooks = {
        PluginType.EPOCH_BEGIN: "start",
        PluginType.EPOCH_END: "stop"
    }

    def __init__(self):
        self.start_time = None

    def start(self, _):
        self.start_time = time.time()

    def stop(self, ctx: TrainContext):
        duration = time.time() - self.start_time
        ctx.workspace['train_time'] = duration
