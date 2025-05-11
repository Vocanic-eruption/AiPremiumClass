from plugins.plugins_base import PluginBase
from trainer.trainer import PluginType, TrainContext


class ChangeLabelShapePlugin(PluginBase):
    plugin_hooks = {
        PluginType.BATCH_BEGIN: "change_shape"
    }
    
    def change_shape(self, ctx: TrainContext):
        ctx.labels = ctx.labels.squeeze(1)
