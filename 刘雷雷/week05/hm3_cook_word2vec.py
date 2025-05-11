# 3. 使用课堂示例cooking.stackexchange.txt，使用fasttext训练文本分类模型。
# （选做：尝试使用Kaggle中的Fake News数据集训练文本分类模型）
# https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

import os
import fasttext

current_dir = os.path.dirname(os.path.abspath(__file__))

modle = fasttext.train_supervised(
    f"{current_dir}/cooking.stackexchange/cooking.stackexchange.txt",
    epoch=100,
    dim=300,
    lr=0.1,
    label_prefix="__label__",
)
print("模型训练完成")

print(modle.predict("What can I use instead of corn syrup?"))
