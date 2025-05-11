import os
import json

from enum import Enum
from .plugins_base import PluginBase
from trainer.trainer import PluginType, TrainContext


class PluginSaveConfig(PluginBase):
    plugin_hooks = {
        PluginType.TRAIN_BEGIN: "save_config"
    }

    def _make_json_safe(self, obj):
        if isinstance(obj, Enum):
            return obj.name
        elif isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_json_safe(v) for v in obj)
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)

    def save_config(self, ctx: TrainContext):
        cfg_dict = ctx.cfg
        cfg_dict = {
            k: v for k, v in cfg_dict.items()
            if not k.startswith('_')
        }
        result_path = ctx.workspace.get("results_folder")

        if result_path is None:
            raise RuntimeError("results_folder not found in ctx.workspace")

        results_folder = result_path
        save_path = os.path.join(results_folder, "config.json")

        safe_cfg = self._make_json_safe(cfg_dict)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(safe_cfg, f, ensure_ascii=False, indent=4)
        
        msg = json.dumps(
            safe_cfg,
            ensure_ascii=False,
            indent=4
        )
        
        if self.check_key(ctx.workspace, "logger"):
            ctx.workspace["logger"](str(msg))
