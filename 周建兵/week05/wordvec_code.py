# import jieba

import fasttext

import os
os.chdir(os.path.dirname(__file__))

# # 分词
# with open('西游记.txt', 'r', encoding='utf-8') as f:
#     text = f.read()

# with open('西游记_cut.txt', 'w', encoding='utf-8') as f:
#     f.write(' '.join(jieba.cut(text)))

model = fasttext.train_unsupervised('xyj_cut.txt', model='skipgram')
# 打印词汇表
# print(len(model.get_words()))    
# 打印词向量
# print(model.get_word_vector('猴子'))

#print(model.get_nearest_neighbors('猴子', k=10))  # 找到与“猴子”最相似的10个词

print(model.get_analogies('猴子', '孙悟空', '唐僧'))  # 找到与“猴子”相关的词

# 保存模型
# model.save_model('fasttext_model.bin')
# 加载模型
# model = fasttext.load_model('fasttext_model.bin')