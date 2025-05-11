# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "./邪王真眼/previous_class09/result",
    "schema_path": "./邪王真眼/datasets/ner/schema.json",
    "train_data_path": "./邪王真眼/datasets/ner/train",
    "valid_data_path": "./邪王真眼/datasets/ner/test",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 3,
    "epoch": 20,
    "batch_size": 8,
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "use_crf": False,
    "class_num": 9,
    "pretrained_model_name": "bert-base-chinese"
}

