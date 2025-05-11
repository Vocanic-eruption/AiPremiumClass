import inspect
import traceback

from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple, Union, Type

from .plugins_base import PluginBase


class PluginManager:
    def __init__(self):
        self.plugins: Dict[str, List[Tuple[int, Callable[[Any], None]]]] = defaultdict(list)
        self.class_plugins: List[PluginBase] = []

    def register(
        self,
        plugin_type: Union[str, None],
        plugin: Union[Callable[[Any], None], Type[PluginBase], PluginBase],
        priority: int = 100,
    ):
        if isinstance(plugin, type) and issubclass(plugin, PluginBase) and hasattr(plugin, "plugin_hooks"):
            instance = plugin()
            self.class_plugins.append(instance)
            self._register_class_plugin(instance, priority)

        elif isinstance(plugin, PluginBase):
            self.class_plugins.append(plugin)
            self._register_class_plugin(plugin, priority)

        elif callable(plugin):
            if plugin_type is None:
                raise ValueError("Function plugin requires explicit plugin_type.")
            self._insert_plugin(plugin_type, plugin, priority)

        else:
            raise TypeError("Unsupported plugin type.")

    def _register_class_plugin(self, plugin: PluginBase, priority: int):
        for hook_type, method_name in plugin.plugin_hooks.items():
            key = hook_type.name if hasattr(hook_type, "name") else str(hook_type)
            
            method = getattr(plugin, method_name, None)
            if not callable(method):
                raise ValueError(f"{plugin.__class__.__name__}.{method_name} is not callable.")

            self._insert_plugin(key, method, priority)

    def _insert_plugin(self, plugin_type: str, plugin_fn: Callable[[Any], None], priority: int):
        current_list = self.plugins[plugin_type]
        new_entry = (priority, plugin_fn)
        for i, (curr_priority, _) in enumerate(current_list):
            if curr_priority > priority:
                current_list.insert(i, new_entry)
                return
        current_list.append(new_entry)
    
    def _get_plugin_info(self, plugin: Callable) -> str:
        if inspect.ismethod(plugin):
            cls_name = plugin.__self__.__class__.__name__
            func_name = plugin.__name__
            return f"{cls_name}.{func_name}"
        elif inspect.isfunction(plugin):
            return plugin.__name__
        elif hasattr(plugin, '__class__'):
            return plugin.__class__.__name__
        else:
            return str(plugin)

    def execute(self, plugin_type: str, context: Any):
        for _, plugin in self.plugins.get(plugin_type, []):
            try:
                plugin(context)
            except Exception as e:
                plugin_info = self._get_plugin_info(plugin)
                print(f"[Plugin Error] Exception in {plugin_info}: {e}")
                traceback.print_exc()
        