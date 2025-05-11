from datasets.datasets import DatasetType
from models.losses.loss_functions import LossFunctionType
from models.optimizers.optimizers import OptimizerType
from models.models import ModelType


config = dict(
    model = ModelType.ResNet,
    num_classes = 10,

    dataset = DatasetType.CIFAR10,
    batch_size = 64,
    data_root = './邪王真眼/datasets/cifar10',

    optimizer = OptimizerType.Adam,
    lr = 0.001,

    loss_function = LossFunctionType.CROSS_ENTROPY,

    epochs = 10,
    result_path = './邪王真眼/previous_class02/result'
)
