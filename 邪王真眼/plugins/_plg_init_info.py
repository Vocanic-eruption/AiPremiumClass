import os
import uuid

from datetime import datetime
from .plugins_base import PluginBase
from trainer.trainer import PluginType, TrainContext


class PluginInitInfo(PluginBase):
    plugin_hooks = {
        PluginType.TRAIN_BEGIN: "initialize"
    }

    def initialize(self, ctx: TrainContext):
        short_uuid = str(uuid.uuid4())[:6]

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_path = ctx.cfg.get("result_path")
        if result_path is None:
            results_folder = os.path.join("./results", f"{timestamp}_{short_uuid}")
        else:
            results_folder = result_path

        ctx.workspace.setdefault("uuid", short_uuid)
        ctx.workspace.setdefault("start_time", timestamp)
        ctx.workspace.setdefault("results_folder", results_folder)

        os.makedirs(ctx.workspace.get("results_folder"), exist_ok=True)
