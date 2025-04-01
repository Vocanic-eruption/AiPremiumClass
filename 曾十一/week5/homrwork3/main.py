# 导入必要的库
import os
import fasttext

# 数据准备
# 假设 cooking.stackexchange.txt 文件已存在且格式正确
# 将数据分为训练集和验证集
# 读取原始数据
with open('/mnt/data_1/zfy/4/week5/homework_3/cooking.stackexchange.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()

# 划分训练集和验证集（例如 80% 训练，20% 验证）
train_ratio = 0.8
train_size = int(len(data) * train_ratio)

train_data = data[:train_size]
val_data = data[train_size:]

# 将划分后的数据写入新的文件
with open('/mnt/data_1/zfy/4/week5/homework_3/cooking_train.txt', 'w', encoding='utf-8') as f:
    for line in train_data:
        f.write(line)

with open('/mnt/data_1/zfy/4/week5/homework_3/cooking_val.txt', 'w', encoding='utf-8') as f:
    for line in val_data:
        f.write(line)

# 模型训练
# 使用 fasttext 的 Python 接口进行训练
model = fasttext.train_supervised(
    input='/mnt/data_1/zfy/4/week5/homework_3/cooking_train.txt',
    epoch=25,
    lr=0.5,
    word_ngrams=2,
    dim=100
)

# 保存模型
model.save_model('/mnt/data_1/zfy/4/week5/homework_3/model_cooking.bin')

# 加载模型
model = fasttext.load_model('/mnt/data_1/zfy/4/week5/homework_3/model_cooking.bin')

# 示例文本
text = ['How much does potato starch affect a cheese sauce recipe?']

# 预测
labels, scores = model.predict(text, k=1)

# 输出结果
print("预测标签:", labels)
print("预测分数:", scores)