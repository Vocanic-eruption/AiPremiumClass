import fasttext_word
import numpy as np
import jieba


# with open('./hlm_c.txt','r',encoding='utf-8') as f:
#     lines = f.read()
#
# with open('./hlm_parse_c.txt','w',encoding='utf-8') as f:
#     f.write(' '.join(jieba.cut(lines)))

model = fasttext.train_unsupervised('./hlm_parse_c.txt',model = 'skipgram')
# print(model.words)
print(len(model.words))
# 获取词向量
print(model.get_word_vector('宝玉'))
# 获取近邻的词
print(model.get_nearest_neighbors('宝玉',k=5))
# 分析词间类比
print(model.get_analogies('宝玉','黛玉','宝钗'))
model.save_model('./hlm_parse_c.bin')
model = fasttext.load_model('./hlm_parse_c.bin')


