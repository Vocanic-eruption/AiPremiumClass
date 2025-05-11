from datasets.datasets import DatasetType


config = dict(
    dataset = DatasetType.E_Commerce_Comments,
    data_root = './邪王真眼/datasets/e_commerce_comments',
    batch_size = 1,

    dim = 200,
    epoch = 50,
    lr = 0.01,
)
