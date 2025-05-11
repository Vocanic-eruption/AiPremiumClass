from datasets.datasets import DatasetType
from models.losses.loss_functions import LossFunctionType
from models.optimizers.optimizers import OptimizerType
from models.models import ModelType


config = dict(
    model = ModelType.RNN,
    rnn_type = 'rnn', # 'rnn' 'lstm' 'gru' 'birnn'
    input_size = 3,
    hidden_size = 128,
    num_classes = 3,
    num_layers = 3,
    dropout = 0.3,

    dataset = DatasetType.Weather,
    batch_size = 512,
    input_days = 50,
    label_days = 1,
    min_station_samples = 1000,
    val_ratio = 0.3,
    data_root = './邪王真眼/datasets/weather',

    optimizer = OptimizerType.Adam,
    lr = 0.0001,

    loss_function = LossFunctionType.MSE,

    epochs = 50,
    result_path = './邪王真眼/week06/result/weather',
)
