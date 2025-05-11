# 1 使用红楼梦文本进行word2vec(cbow skip_gram两种模型)的词向量的训练
# 2 使用cooking数据进行文本分类任务

import jieba
import torch.nn as nn
import numpy as np
import csv
import fasttext #fasttext 的使用必须是分词后的文件

# 1 word2vec
#文件读取与预处理分词
with open('hlm_c.txt','r',encoding = 'utf-8') as f:
    lines = f.read() # f.read() 读入的是整个文本 的一个大列表 以字为单位
    #print(lines[:10])

with open('hlm_c_sphase.txt','w',encoding = 'utf-8') as f:
    f.write(' '.join(jieba.lcut(lines)))

#构建模型
model = fasttext.train_unsupervised('hlm_c_sphase.txt',model = 'skipgram') #无监督 是word2vec模型的词向量训练  默认词向量维度是100
#print('文档词汇表',model.words) #把文档里的所有词汇都构建出来 并且也训练出了一组词向量 是个list

#查看一个词训练出来的词向量
print(model.get_word_vector('黛玉'))

#获取文章词向量的最近邻词 捕获文章中哪些词和这个词组合出现的次数最高
print(model.get_nearest_neighbors('雪',k = 5))

#词汇间的对比 分析几个词之间到底有什么关系 之间的关系距离到底是都多少
print(model.get_analogies('贾宝玉','林黛玉','薛宝钗'))
#以上就是某种推荐系统的基础
#这种模型把我们的自然语言和神经网络联系起来 为后来的人工智能发展打下重要基础

# #模型的保存
# model.save_model('hlm_skipgram.bin')
# #加载模型
# model = fasttext.load_model('hlm_skipgram.bin')





