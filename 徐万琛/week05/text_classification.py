# 使用fasttext训练文本分类模型
# 使用cooking.stackexchange.txt和Fake News数据集

import os
import re
import pandas as pd
import numpy as np
import fasttext
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# 设置文件路径
base_path = os.path.dirname(os.path.abspath(__file__))
cooking_file = os.path.join(base_path, 'cooking.stackexchange.txt')
fake_news_file = os.path.join(base_path, 'Fake_News', 'WELFake_Dataset.csv')

# 数据预处理函数
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 去除URL
    text = re.sub(r'http\S+', '', text)
    # 去除特殊字符
    text = re.sub(r'[^\w\s]', ' ', text)
    # 转换为小写
    text = text.lower()
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 准备cooking.stackexchange.txt数据
def prepare_cooking_data():
    print("准备cooking.stackexchange.txt数据...")
    
    # 读取数据
    with open(cooking_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 预处理数据
    processed_lines = []
    for line in lines:
        # 提取标签和文本
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
            text = parts[1]
            # 提取标签
            labels = [label for label in parts[0].split() if label.startswith('__label__')]
            # 预处理文本
            text = preprocess_text(text)
            # 添加到处理后的数据中
            if text and labels:
                processed_line = ' '.join(labels) + ' ' + text
                processed_lines.append(processed_line)
    
    # 划分训练集和测试集
    train_data, test_data = train_test_split(processed_lines, test_size=0.2, random_state=42)
    
    # 保存为fasttext格式的文件
    train_file = os.path.join(base_path, 'cooking_train.txt')
    test_file = os.path.join(base_path, 'cooking_test.txt')
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_data))
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_data))
    
    return train_file, test_file

# 训练cooking.stackexchange.txt分类模型
def train_cooking_model(train_file, test_file):
    print("训练cooking.stackexchange.txt分类模型...")
    
    # 训练模型
    model = fasttext.train_supervised(
        train_file,
        lr=0.1,
        epoch=25,
        wordNgrams=2,
        bucket=200000,
        dim=100,
        loss='ova'
    )
    
    # 评估模型
    result = model.test(test_file)
    print(f"测试集准确率: {result[1]:.4f}")
    print(f"测试集召回率: {result[2]:.4f}")
    
    # 保存模型
    model_file = os.path.join(base_path, 'cooking_model.bin')
    model.save_model(model_file)
    
    # 返回模型和测试文件
    return model, test_file

# 可视化cooking模型结果
def visualize_cooking_results(model, test_file):
    print("可视化cooking模型结果...")
    
    # 读取测试数据
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = f.readlines()
    
    # 提取真实标签和文本
    true_labels = []
    texts = []
    for line in test_data:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
            # 提取第一个标签作为主要标签
            label = parts[0].split()[0].replace('__label__', '')
            true_labels.append(label)
            texts.append(parts[1])
    
    # 预测标签
    pred_labels = []
    for text in texts:
        # 获取预测结果
        prediction = model.predict(text, k=1)
        # 提取预测标签
        pred_label = prediction[0][0].replace('__label__', '')
        pred_labels.append(pred_label)
    
    # 计算评估指标
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 统计标签分布
    unique_labels = list(set(true_labels))
    if len(unique_labels) <= 10:  # 只在标签数量较少时绘制混淆矩阵
        # 计算混淆矩阵
        cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('Cooking分类模型混淆矩阵')
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, 'cooking_confusion_matrix.png'))
        plt.close()

# 准备Fake News数据（选做部分）
def prepare_fake_news_data():
    print("准备Fake News数据...")
    
    try:
        # 读取数据
        df = pd.read_csv(fake_news_file)
        
        # 检查数据结构
        print(f"Fake News数据集形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        
        # 假设数据集包含'title', 'text'和'label'列
        # 如果列名不同，需要根据实际情况调整
        if 'title' in df.columns and 'text' in df.columns and 'label' in df.columns:
            # 合并标题和正文
            df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
            
            # 预处理文本
            df['processed_content'] = df['content'].apply(preprocess_text)
            
            # 转换标签格式为fasttext格式
            df['fasttext_label'] = '__label__' + df['label'].astype(str)
            
            # 创建fasttext格式的数据
            fasttext_data = df['fasttext_label'] + ' ' + df['processed_content']
            
            # 划分训练集和测试集
            train_data, test_data = train_test_split(fasttext_data, test_size=0.2, random_state=42)
            
            # 保存为fasttext格式的文件
            train_file = os.path.join(base_path, 'fake_news_train.txt')
            test_file = os.path.join(base_path, 'fake_news_test.txt')
            
            with open(train_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(train_data))
            
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(test_data))
            
            return train_file, test_file
        else:
            print("Fake News数据集结构不符合预期，无法处理")
            return None, None
    except Exception as e:
        print(f"处理Fake News数据时出错: {e}")
        return None, None

# 训练Fake News分类模型（选做部分）
def train_fake_news_model(train_file, test_file):
    if not train_file or not test_file:
        print("Fake News数据准备失败，跳过模型训练")
        return None, None
    
    print("训练Fake News分类模型...")
    
    # 训练模型
    model = fasttext.train_supervised(
        train_file,
        lr=0.1,
        epoch=25,
        wordNgrams=2,
        bucket=200000,
        dim=100,
        loss='softmax'
    )
    
    # 评估模型
    result = model.test(test_file)
    print(f"测试集准确率: {result[1]:.4f}")
    print(f"测试集召回率: {result[2]:.4f}")
    
    # 保存模型
    model_file = os.path.join(base_path, 'fake_news_model.bin')
    model.save_model(model_file)
    
    # 返回模型和测试文件
    return model, test_file

# 可视化Fake News模型结果（选做部分）
def visualize_fake_news_results(model, test_file):
    if not model or not test_file:
        print("Fake News模型训练失败，跳过可视化")
        return
    
    print("可视化Fake News模型结果...")
    
    # 读取测试数据
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = f.readlines()
    
    # 提取真实标签和文本
    true_labels = []
    texts = []
    for line in test_data:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
            label = parts[0].replace('__label__', '')
            true_labels.append(label)
            texts.append(parts[1])
    
    # 预测标签
    pred_labels = []
    for text in texts:
        prediction = model.predict(text, k=1)
        pred_label = prediction[0][0].replace('__label__', '')
        pred_labels.append(pred_label)
    
    # 计算评估指标
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 计算混淆矩阵
    unique_labels = list(set(true_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('Fake News分类模型混淆矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'fake_news_confusion_matrix.png'))
    plt.close()
    
    # 绘制ROC曲线（如果是二分类问题）
    if len(unique_labels) == 2:
        from sklearn.metrics import roc_curve, auc
        
        # 获取预测概率
        pred_probs = []
        for text in texts:
            prediction = model.predict(text, k=1)
            prob = prediction[1][0]  # 获取预测概率
            pred_probs.append(prob)
        
        # 转换标签为数值
        true_labels_num = [1 if label == unique_labels[1] else 0 for label in true_labels]
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(true_labels_num, pred_probs)
        roc_auc = auc(fpr, tpr)
        
        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('Fake News分类模型ROC曲线')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(base_path, 'fake_news_roc_curve.png'))
        plt.close()

# 主函数
def main():
    print("开始文本分类任务...")
    
    # 处理cooking.stackexchange.txt数据
    cooking_train_file, cooking_test_file = prepare_cooking_data()
    cooking_model, cooking_test_file = train_cooking_model(cooking_train_file, cooking_test_file)
    visualize_cooking_results(cooking_model, cooking_test_file)
    
    # 处理Fake News数据（选做部分）
    print("\n开始处理Fake News数据（选做部分）...")
    fake_news_train_file, fake_news_test_file = prepare_fake_news_data()
    fake_news_model, fake_news_test_file = train_fake_news_model(fake_news_train_file, fake_news_test_file)
    visualize_fake_news_results(fake_news_model, fake_news_test_file)
    
    print("\n文本分类任务完成！")

if __name__ == "__main__":
    main()