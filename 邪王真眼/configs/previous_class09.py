from datasets.datasets import DatasetType
from models.optimizers.optimizers import OptimizerType
from models.models import ModelType


config = dict(
    model = ModelType.BertClassifier,
    input_size = 128,
    hidden_size = 256,
    num_layers = 3,
    dropout = 0.1,

    dataset = DatasetType.NER,
    batch_size = 8,
    data_root = './邪王真眼/datasets/corpus_p9',
    max_length = 100,

    optimizer = OptimizerType.Adam,
    lr = 0.0001,

    epochs = 10,
    result_path = './邪王真眼/previous_class09/result'
)
