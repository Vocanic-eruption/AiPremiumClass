
import fasttext 
import jieba


# 文档预处理
# with open('xyj.txt','r',encoding='utf-8') as f:
#     lines = f.read()

 
# with open('xyj_sparse_c.txt','w',encoding='utf-8') as f:
#    f.write(' '.join(jieba.lcut(lines)))
# model = fasttext.train_unsupervised('xyj_sparse_c.txt',model='skipgram', dim=300,epoch=200,wordNgrams=7)
model = fasttext.load_model('xyj_skipgram.bin')
print("文档词汇表长度",len(model.words))

# 获取词向量
print('词向量:',model.get_word_vector('孙悟空'))

# 获取近邻词
print('近邻词',model.get_nearest_neighbors('孙悟空',k=5))

# 分析词汇间类比
print("词汇间类比",model.get_analogies('孙悟空','猪八戒','沙僧'))

#  保存模型
#model.save_model('xyj_skipgram.bin')
# 加载模型
# model = fasttext.load_model('xyj_skipgram.bin')