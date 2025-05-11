from abc import ABC
from typing import Dict


class PluginBase(ABC):
    plugin_hooks: Dict = {}
    _warned_keys = set()

    def __init__(self):
        pass

    def check_key(self, ctx: dict, key: str, warn_msg: str = None):
        if key not in ctx and key not in self._warned_keys:
            if warn_msg is not None:
                print(f"{warn_msg}")
            self._warned_keys.add(key)
        
        return key in ctx
