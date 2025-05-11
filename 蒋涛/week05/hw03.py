"""
3. 使用课堂示例cooking.stackexchange.txt，使用fasttext训练文本分类模型。（选做：尝试使用Kaggle中的Fake News数据集训练文本分类模型）
https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
"""
import fasttext
import pandas as pd
from sklearn.model_selection import train_test_split


def train_and_evaluate():

    # 训练模型
    model = fasttext.train_supervised(input='/Users/jiangtao/PycharmProjects/study-ai/week5_语言模型及词向量相关知识/cooking.stackexchange.txt')

    # 保存模型
    model.save_model("cooking_model.bin")

    # 加载模型
    loaded_model = fasttext.load_model("cooking_model.bin")

    # 示例预测
    text = "How do I cover up the white spots on my cast iron stove?"
    predicted_label = loaded_model.predict(text)
    print(f"预测标签: {predicted_label}")

    # 评估模型
    result = loaded_model.test('/Users/jiangtao/PycharmProjects/study-ai/week5_语言模型及词向量相关知识/cooking.stackexchange.txt')
    print(f"样本数量: {result[0]}, 准确率: {result[1]}, 召回率: {result[2]}")

# 读取数据集
data = pd.read_csv('/Users/jiangtao/Downloads/WELFake_Dataset.csv/WELFake_Dataset.csv')  # 替换为实际的数据集文件名

# 数据预处理，将数据集转换为 fasttext 所需的格式
def prepare_data(data, label_column, text_column):
    data['label'] = data[label_column].apply(lambda x: f'__label__{x}')
    data['fasttext_data'] = data['label'] + ' ' + data[text_column]
    return data['fasttext_data']

def train_and_evaluate_fake():
    # 读取数据集
    data = pd.read_csv('/Users/jiangtao/Downloads/WELFake_Dataset.csv/WELFake_Dataset.csv')  # 替换为实际的数据集文件名
    # 假设数据集中有 'label' 和 'text' 两列
    fasttext_data = prepare_data(data, 'label', 'text')

    # 划分训练集和测试集
    train_data, test_data = train_test_split(fasttext_data, test_size=0.2, random_state=42)

    # 保存训练集和测试集到文件
    train_data.to_csv('WELFake_train.txt', index=False, header=False)
    test_data.to_csv('WELFake_test.txt', index=False, header=False)

    # 训练模型
    model = fasttext.train_supervised(input='WELFake_train.txt')

    # 保存模型
    model.save_model("WELFake_news_model.bin")

    # 加载模型
    loaded_model = fasttext.load_model("WELFake_news_model.bin")

    # 示例预测
    text = "This is a sample news article."
    predicted_label = loaded_model.predict(text)
    print(f"预测标签: {predicted_label}")

    # 评估模型
    result = loaded_model.test('WELFake_test.txt')
    print(f"样本数量: {result[0]}, 准确率: {result[1]}, 召回率: {result[2]}")

if __name__ == "__main__":
    train_and_evaluate()
    train_and_evaluate_fake()