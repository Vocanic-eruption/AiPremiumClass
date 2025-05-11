import fasttext
import jieba

#文档处理
# with open("sgyy.txt","r",encoding="utf-8") as f:
#     lines = f.read()
# with open('sgyy_space.txt','w',encoding='utf-8') as f:
#     f.write(' '.join(jieba.lcut(lines)))

model_skipgram = fasttext.train_unsupervised('sgyy_space.txt', model='skipgram')
model_cbow = fasttext.train_unsupervised('sgyy_space.txt', model='cbow')

#模型加载
# model_skipgram = fasttext.load_model('model_skipgram.bin')
# model_cbow = fasttext.load_model('model_cbow.bin')

#获取词向量
print(model_skipgram.get_word_vector('关羽'))
print(model_cbow.get_word_vector('关羽'))

#获取近邻词
print(model_skipgram.get_nearest_neighbors('关羽'))
print(model_cbow.get_nearest_neighbors('关羽'))

#词汇间类比
print(model_skipgram.get_analogies('关羽','刘备','张飞'))
print(model_cbow.get_analogies('关羽','刘备','张飞'))

# #模型保存
# model_skipgram.save_model('model_skipgram.bin')
# model_cbow.save_model('model_cbow.bin')
