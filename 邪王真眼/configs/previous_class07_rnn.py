from datasets.datasets import DatasetType
from models.losses.loss_functions import LossFunctionType
from models.optimizers.optimizers import OptimizerType
from models.models import ModelType


config = dict(
    model = ModelType.RNN,
    rnn_type = 'gru',
    num_classes = 2,
    input_size = 64,
    hidden_size = 64,
    num_layers = 2,
    dropout = 0.1,

    dataset = DatasetType.E_Commerce_Comments_Idx,
    batch_size = 512,
    data_root = './邪王真眼/datasets/e_commerce_comments',
    max_length = 128,

    optimizer = OptimizerType.Adam,
    lr = 0.01,

    loss_function = LossFunctionType.BCE_WITH_LOGITS,

    epochs = 20,
    result_path = './邪王真眼/previous_class07/result'
)
