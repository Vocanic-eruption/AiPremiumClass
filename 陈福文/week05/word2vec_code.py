
# import jieba

# with open("D://ai//badou//codes//第五周//hlm.txt","r",encoding="utf-8") as f:
#     lines = f.read()

# with open("D://ai//badou//codes//第五周//hlm_cut.txt","w",encoding="utf-8") as f:
#     f.write(" ".join(jieba.cut(lines)))

import fasttext
# with open("D://ai//badou//codes//第五周//hlm_cut.txt","r",encoding="utf-8") as f:
#     lines = f.readlines()

model = fasttext.train_unsupervised("D:\\ai\\badou\\codes\\week05\\hlm_cut.txt", model='skipgram')
# model = fasttext.train_unsupervised("D://ai//badou//codes//第五周//hlm.txt", model='cbow')

 # 保存模型
model.save_model('hongloumeng.bin')

# 加载模型
model = fasttext.load_model('hongloumeng.bin')

print('计算词汇间的相关度', '\n', model.get_nearest_neighbors('名唤',k=10))
