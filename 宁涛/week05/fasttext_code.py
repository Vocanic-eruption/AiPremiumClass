import fasttext

model = fasttext.train_supervised('cooking.stackexchange.txt', epoch=10, dim=200)

print(model.predict("What kind of tea do you boil for 45minutes?"))
#获取词向量
model.get_word_vector('Water')
#获取近邻词
model.get_nearest_neighbors('Water', k=5)
#分析词间类比
model.get_analogies('Water','Beer')
