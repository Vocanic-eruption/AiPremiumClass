from datasets.datasets import DatasetType
from models.losses.loss_functions import LossFunctionType
from models.optimizers.optimizers import OptimizerType
from models.models import ModelType


config = dict(
    model = ModelType.RNN,
    rnn_type = 'rnn',
    input_size = 128,
    hidden_size = 128,
    num_layers = 2,
    dropout = 0.1,

    dataset = DatasetType.Corpus_P6,
    batch_size = 128,
    data_root = './邪王真眼/datasets/corpus_p6',
    window_length = 6,
    sample_length = 10000,

    optimizer = OptimizerType.Adam,
    lr = 0.001,

    loss_function = LossFunctionType.CROSS_ENTROPY,

    epochs = 10,
    result_path = './邪王真眼/previous_class06/result'
)
