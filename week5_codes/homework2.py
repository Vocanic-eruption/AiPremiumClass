import fasttext
import jieba
import torch
from torch.utils.tensorboard import SummaryWriter

# # 文档分词器
# tokenizer = lambda x: jieba.lcut(x)

# # 加载原文档
# with open('流浪地球.txt', 'r', encoding='utf-8') as f:
#     text = f.read()
# # 分词
# words = tokenizer(text)
# # 保存分词后的文档
# with open('流浪地球_t.txt', 'w', encoding='utf-8') as f:
#     for word in words:
#         f.write(word + ' ')

model = fasttext.train_unsupervised('流浪地球_t.txt', epoch=20, dim=300)

# word2vec模型使用方法，文本分类中一样可用
print('分析"太阳" 近似词：', model.get_nearest_neighbors('太阳'))
print('分析"地球" 近似词：', model.get_nearest_neighbors('地球'))
print('分析"流浪" 近似词：', model.get_nearest_neighbors('流浪'))

# 选做：词向量可视化
writer = SummaryWriter()
meta_data = model.words
embeddings = []
for word in meta_data:
    embeddings.append(model.get_word_vector(word))

writer.add_embedding(torch.tensor(embeddings), metadata=meta_data)