from datasets.datasets import DatasetType
from models.losses.loss_functions import LossFunctionType
from models.optimizers.optimizers import OptimizerType
from models.models import ModelType


config = dict(
    model = ModelType.CBOW,
    embedding_dim = 256,

    dataset = DatasetType.CBOW,
    batch_size = 1024,
    data_root = './邪王真眼/datasets/word2vec',
    use_lines = 2000,
    window_length = 2,

    optimizer = OptimizerType.Adam,
    lr = 0.01,

    loss_function = LossFunctionType.CROSS_ENTROPY,

    epochs = 20,
    result_path = './邪王真眼/previous_class05/result/cbow'
)
