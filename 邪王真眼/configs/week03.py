from datasets.datasets import DatasetType
from models.losses.loss_functions import LossFunctionType
from models.optimizers.optimizers import OptimizerType
from models.models import ModelType


config = dict(
    model = ModelType.CNN,
    input_size = (28, 28),
    input_channels = 1,
    hidden_channels = 64,

    dataset = DatasetType.KMNIST,
    batch_size = 512,
    data_root = './邪王真眼/datasets/kmnist',

    optimizer = OptimizerType.Adam,
    num_classes = 10,
    lr = 0.01,

    loss_function = LossFunctionType.CROSS_ENTROPY,

    epochs = 10,
    result_path = './邪王真眼/week03/result',
)
