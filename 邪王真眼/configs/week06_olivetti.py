from datasets.datasets import DatasetType
from models.losses.loss_functions import LossFunctionType
from models.optimizers.optimizers import OptimizerType
from models.models import ModelType


config = dict(
    model = ModelType.RNN,
    rnn_type = 'rnn', # 'rnn' 'lstm' 'gru' 'birnn'
    input_size = 64,
    hidden_size = 256,
    num_classes = 40,
    num_layers = 3,
    dropout = 0.3,

    dataset = DatasetType.OLIVETTI_FACES,
    batch_size = 4,
    data_root = './邪王真眼/datasets/olivetti_faces',

    optimizer = OptimizerType.Adam,
    lr = 0.0001,

    loss_function = LossFunctionType.CROSS_ENTROPY,

    epochs = 100,
    result_path = './邪王真眼/week06/result/olivetti',
)
