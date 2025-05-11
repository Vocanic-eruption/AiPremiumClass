# 2. 使用自定义的文档文本，通过fasttext训练word2vec训练词向量模型，
# 并计算词汇间的相关度。
# (选做：尝试tensorboard绘制词向量可视化图）

import os
import fasttext
from torch.utils.tensorboard import SummaryWriter
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))

# 模型
model = fasttext.train_unsupervised(
    f"{current_dir}/hlm_sprase.txt", epoch=100, dim=300, lr=0.1
)

# word2vec模型使用方法
print("文档词汇表长度:", len(model.words))
print("分析 宝玉 近似词:", model.get_nearest_neighbors("宝玉"))
print("分析 黛玉 近似词:", model.get_nearest_neighbors("黛玉"))
print("分析 宝钗 近似词:", model.get_nearest_neighbors("宝钗"))

# # 词限量可视化
print("词向量可视化开始")
writer = SummaryWriter()
meta_data = model.words

embedding = []
for word in meta_data:
    embedding.append(model.get_word_vector(word))

# 词向量可视化
writer.add_embedding(torch.tensor(embedding), metadata=meta_data)
writer.close()
print("词向量可视化结束")
