import jieba
import fasttext
import os
file_path = "e:\\workspacepython\\AiPremiumClass\\肖路平\\week5\\hlm_s.txt"
print("文件是否存在:", os.path.exists(file_path))
print("是否可读:", os.access(file_path, os.R_OK))
 
#model = fasttext.train_unsupervised(r"e:\workspacepython\AiPremiumClass\肖路平\week5\hlm_sprase_c.txt", model='skipgram')

model = fasttext.train_unsupervised("e:\\workspacepython\\AiPremiumClass\\肖路平\\week5\\hlm_s.txt", model='skipgram')

print('文档词汇表：',model.words)

# 获取词向量
print(model.get_word_vector('宝玉'))

#获取紧邻词
print(model.get_nearest_neighbors('宝玉',k=5))

#分析词间类比
print(model.get_analogies('宝玉','黛玉','宝钗'))
