from models.models import ModelType
from utils.tokenizer import SegmenterType
from datasets.datasets import DatasetType
from models.simple_attention import MergeMode
from models.optimizers.optimizers import OptimizerType
from models.losses.loss_functions import LossFunctionType


config = dict(
    model = ModelType.SIMPLE_ATTENTION,
    embedding_dim = 128,
    hidden_size = 128,
    merge_mode = MergeMode.SUM,
    
    dataset = DatasetType.COUPLET,
    batch_size = 256,
    max_len = 30,
    num_data = 100000,
    data_root = './邪王真眼/datasets/couplet',

    optimizer = OptimizerType.Adam,
    lr = 0.01,
    momentum = 0.9,
    weight_decay = 0.0001,

    loss_function = LossFunctionType.CROSS_ENTROPY,
    ignore_index = 0,
    
    epochs = 10,
    result_path = './邪王真眼/week08/result'
)
