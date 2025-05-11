from datasets.datasets import DatasetType
from models.losses.loss_functions import LossFunctionType
from models.optimizers.optimizers import OptimizerType
from models.models import ModelType


config = dict(
    model = ModelType.Logistic_Regression,
    input_size = 30,

    dataset = DatasetType.BREAST_CANCER,
    batch_size = 32,

    optimizer = OptimizerType.SGD,
    num_classes = 2,
    lr = 0.01,

    loss_function = LossFunctionType.CROSS_ENTROPY,

    epochs = 30,
    result_path = './邪王真眼/week02/result'
)
