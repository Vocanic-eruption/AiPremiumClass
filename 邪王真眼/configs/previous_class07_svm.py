from datasets.datasets import DatasetType


config = dict(
    dataset = DatasetType.E_Commerce_Comments,
    data_root = './邪王真眼/datasets/e_commerce_comments',
    batch_size = 1,

    ngram_range = (1, 1),
    c = 10,
    kernel = 'linear', # 'rbf'
)
