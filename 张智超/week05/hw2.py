import fasttext

model_cbow = fasttext.train_unsupervised('./data/fixed/three_body_fixed.txt', model='cbow')
model_skipgram = fasttext.train_unsupervised('./data/fixed/three_body_fixed.txt', model='skipgram')

# 获取词向量的最近邻
n1 = model_cbow.get_nearest_neighbors('叶文洁')
print(n1)
print('=====================')
n2 = model_skipgram.get_nearest_neighbors('叶文洁')
print(n2)


