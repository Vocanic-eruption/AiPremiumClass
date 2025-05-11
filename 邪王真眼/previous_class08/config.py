Config = {
    "model_path": "./邪王真眼/previous_class08/result",
    "schema_path": "./邪王真眼/datasets/corpus_p8/schema.json",
    "vocab_path":"./邪王真眼/datasets/corpus_p8/chars.txt",

    "train_data_path": "./邪王真眼/datasets/corpus_p8/train.json",
    "valid_data_path": "./邪王真眼/datasets/corpus_p8/valid.json",
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 10,
    "batch_size": 32,
    "epoch_data_size": 200,
    "positive_sample_rate":0.5,
    "optimizer": "adam",
    "learning_rate": 1e-3,
}
