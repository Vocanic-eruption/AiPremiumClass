from datasets.datasets import DatasetType
from models.losses.loss_functions import LossFunctionType
from models.optimizers.optimizers import OptimizerType
from models.models import ModelType
from utils.tokenizer import SegmenterType

config = dict(
    model = ModelType.RNN,
    num_classes = 2,
    rnn_type = 'lstm',
    hidden_size = 256,
    input_size = 256,
    num_layers = 2,
    dropout = 0.3,

    dataset = DatasetType.DOUBAN_COMMENTS,
    split_ratio = 0.7,
    batch_size = 512,
    data_max_num = 5000,
    seg_type = SegmenterType.SENTENCE_PIECE,
    data_root = './邪王真眼/datasets/douban_comments',

    optimizer = OptimizerType.SGD,
    lr = 0.01,
    momentum = 0.9,
    weight_decay = 0.0001,

    loss_function = LossFunctionType.CROSS_ENTROPY,

    epochs = 10,
    result_path = './邪王真眼/week07/result'
)
