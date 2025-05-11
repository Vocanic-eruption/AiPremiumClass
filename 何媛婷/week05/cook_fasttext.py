import fasttext
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

"""
关键说明
‌多标签处理‌：cooking.stackexchange.txt 支持多标签分类（如 __label__baking __label__equipment），需设置 loss='ova'（one-vs-all）。Fake News 是二分类问题（fake/real），默认使用 loss='softmax'。
‌性能优化‌：调整 lr（学习率）、epoch（训练轮次）、wordNgrams（词组合特征）以平衡速度与精度。
文本清洗（如去除特殊字符、小写化）可显著提升模型鲁棒性。
‌文件路径‌：
确保代码中的文件路径（如 cooking_train.txt）与实际存储位置一致。
‌扩展性‌：
支持自定义标签体系（需保持训练/预测时标签格式一致）。
"""

def fetch_model(datafilepath,lr,epoch,wordNgrams):
    """
    模型训练
    """
    model = fasttext.train_supervised(
        input=datafilepath,
        lr=lr,
        epoch=epoch,
        wordNgrams=wordNgrams
    )
    return model

def split_data(datafilepath):
    """
    读取数据并拆分
    """
    with open(datafilepath, "r", encoding="utf-8") as f:
        data = f.readlines()
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    # 保存训练集和测试集
    with open("./data/cooking_train.txt", "w", encoding="utf-8") as f:
        f.writelines(train_data)
    with open("./data/cooking_test.txt", "w", encoding="utf-8") as f:
        f.writelines(test_data)
    return train_data,test_data


if __name__=='__main__':
    # datafilepath = "./data/cooking.stackexchange.txt"
    # 读取数据并拆分
    # train_data,test_data = split_data(datafilepath)
    # 模型训练
    model = fetch_model("./data/cooking_train.txt",0.5,50,2)

    # 使用 fastText 内置评估
    precision, recall, _ = model.test("./data/cooking_test.txt")
    print(f"准确率: {precision*100:.2f}%")
    print(f"召回率: {recall*100:.2f}%")


    # 预测新文本
    text = "如何正确打发蛋白？"
    labels, scores = model.predict(text, k=2)  # 返回前 2 个预测标签
    print(f"输入文本: {text}")
    print(f"预测标签: {labels} (置信度: {scores})")





