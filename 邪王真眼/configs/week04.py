from datasets.datasets import DatasetType
from models.losses.loss_functions import LossFunctionType
from models.optimizers.optimizers import OptimizerType
from models.models import ModelType


config = dict(
    model = ModelType.CNN,
    input_size = (64, 64),
    input_channels = 1,
    hidden_channels = 128,

    dataset = DatasetType.OLIVETTI_FACES,
    batch_size = 4,
    data_root = './邪王真眼/datasets/olivetti_faces',

    optimizer = OptimizerType.AdamW,
    num_classes = 40,
    lr = 0.0001,
    weight_decay = 1e-4,

    loss_function = LossFunctionType.CROSS_ENTROPY,

    epochs = 20,
    result_path = './邪王真眼/week04/result',
)
