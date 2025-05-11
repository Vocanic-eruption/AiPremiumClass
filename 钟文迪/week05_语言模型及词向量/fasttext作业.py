import fasttext
import torch
from torch.utils.tensorboard import SummaryWriter

# Skipgram model
model = fasttext.train_unsupervised('./data.txt', model='skipgram', minCount=1)

# cbow model
# model = fasttext.train_unsupervised('./data.txt', model='cbow', minCount=1)

# 获取词向量的最近邻
print(model.get_nearest_neighbors('helloworld'))

# 词汇间类比
print(model.get_analogies('he', 'lo', 'l'))

# 保存模型
# model.save_model('model.bin')

# # 加载模型
# word_vectors = fasttext.load_model('model.bin')
# print(word_vectors.words[:3])

# writer = SummaryWriter()
# meta = []
# while len(meta) < 3:
#     i = len(meta)
#     meta.append(word_vectors.words[i]) # collect word list
# meta = meta[:3]

# writer.add_embedding(torch.tensor(word_vectors.words[:3]), metadata=meta)