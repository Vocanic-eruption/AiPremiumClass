import os
import logging
import requests
import threading

from .plugins_base import PluginBase
from trainer.trainer import PluginType, TrainContext


class PluginLogger(PluginBase):
    plugin_hooks = {
        PluginType.TRAIN_BEGIN: "setup"
    }

    def setup(self, ctx: TrainContext):
        result_path = ctx.cfg.get("result_path")
        if result_path is None:
            results_folder = os.path.join("./results")
        else:
            results_folder = result_path
        
        results_folder = ctx.workspace.setdefault("results_folder", results_folder)
        os.makedirs(results_folder, exist_ok=True)
        log_path = os.path.join(results_folder, "log.txt")

        self.logger = logging.getLogger(f"{ctx.workspace.get('uuid', 'any')}")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(fh)

        def log_fn(message: str, print_only: bool = False):
            if print_only:
                print(message, end='', flush=True)
                return
            else:
                print(message)

            self.logger.info(message)

            payload = {
                "title": ctx.workspace.get("uuid", "trainer"),
                "text": message
            }
            threading.Thread(
                target=requests.post,
                args=("http://47.115.218.122:5149/api/send_message",),
                kwargs={"json": payload},
                daemon=True
            ).start()

        ctx.workspace["logger"] = log_fn
