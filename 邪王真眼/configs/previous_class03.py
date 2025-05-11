from datasets.datasets import DatasetType
from models.losses.loss_functions import LossFunctionType
from models.optimizers.optimizers import OptimizerType
from models.models import ModelType


config = dict(
    model = ModelType.RNN,
    rnn_type = 'gru',
    input_size = 64,
    hidden_size = 64,
    num_classes = 11,
    num_layers = 2,
    vocab_size = 30522,

    dataset = DatasetType.SEM_EVAL,
    batch_size = 32,
    data_root = './邪王真眼/datasets/sem_eval',

    optimizer = OptimizerType.Adam,
    lr = 0.01,

    loss_function = LossFunctionType.BCE_WITH_LOGITS,

    epochs = 10,
    result_path = './邪王真眼/previous_class03/result'
)
