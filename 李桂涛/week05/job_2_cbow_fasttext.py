import fasttext
import torch
from torch.utils.tensorboard import SummaryWriter

#train_unsupervised方法来训练无监督的词向量模型。模型设置cbow（连续词袋模型）也可以用skipgram（跳跃词袋模型）
#model = fasttext.train_unsupervised('cbow.txt', model='skipgram')
#1、无监督训练词向量模型
model = fasttext.train_unsupervised('cbow.txt', model='cbow', minCount=1)
#2、保存模型
model.save_model('cbow_model.bin')
#3、加载模型
fasttext.load_model('cbow_model.bin')

#测试:
# 计算单词的相似度
print('分析"小狗" 近似词：', model.get_nearest_neighbors('dog'))

#使用tensorboard 进行绘图
writer = SummaryWriter()
meta_data = model.words
embeddings = []
for word in meta_data:
    embeddings.append(model.get_word_vector(word))

writer.add_embedding(torch.tensor(embeddings), metadata=meta_data)

# # 获取单词的词向量
# vector = model.get_word_vector('fox')
# print(vector)

