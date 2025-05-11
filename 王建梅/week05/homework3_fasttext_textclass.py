import os
import fasttext
import pandas as pd
import csv
import sys

# 增加 CSV 字段大小限制
csv.field_size_limit(2147483647)  # 设置为一个较大的值，如 2GB

# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
cooking_txt_path = os.path.join(current_dir, 'cooking.stackexchange.txt')
welfake_csv_file = os.path.join(current_dir, 'WELFake_Dataset.csv')
welfake_txt_file = os.path.join(current_dir, 'welfake.txt')

# csv文件转换为txt文件
def welfake_dataset(input_path,output_path):
    with open(input_path, 'r', encoding='utf-8') as csvfile,open(output_path, 'w', encoding='utf-8') as txtfile:
        # 读取csv文件
        reader = csv.DictReader(csvfile)
        for row in reader:
                label = row['label']
                text = row['text']
                # 将标签和文本写入到txt文件中
                txtfile.write(f"__label__{label} {text}\n")

def textclassification(model,text):
    print(f"Text: {text}")
    labels, probabilities = model.predict(text)
    print(f"预测标签: {labels}, 预测概率: {probabilities}")

def train_model(file_path):
    # 训练模型 有监督训练，每行数据需要包含一个或多个以 __label__ 开头的标签，后面跟着对应的文本内容，标签和文本之间用空格分隔
    model = fasttext.train_supervised(file_path, epoch=10)
    return model

if __name__ == "__main__":
    # 将csv文件转换为txt文件
    welfake_dataset(welfake_csv_file,welfake_txt_file)
    # Kaggle中的Fake News数据集训练文本分类模型  https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
    cooking_model = train_model(cooking_txt_path)
    wlefake_model = train_model(welfake_txt_file)
    # 测试文本分类
    textclassification(cooking_model,"What is the best way to cook a steak?")
    textclassification(wlefake_model,"This is a fake news.")
    

    