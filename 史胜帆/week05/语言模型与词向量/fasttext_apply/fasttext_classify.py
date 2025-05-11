# 1 使用红楼梦文本进行word2vec(cbow skip_gram两种模型)的词向量的训练
# 2 使用cooking数据进行文本分类任务

import fasttext #比pytorch训练快 因为fasttext的底层是用C++写的



# 2 fasttext 的文本分类功能 有监督学习  数据需要人工标注分类
#还是做了词向量的训练
model = fasttext.train_supervised('cooking.stackexchange.txt')

#除了具备word2vec所具有的k近邻、词汇类比、打印词向量  还具备文本分类的任务
print(model.predict('What kind of tea do you boil for 45minutes?'))

#增加模型参数 提高训练效果和准确性
#model = fasttext.train_supervised('cooking.stackexchange.txt',epoch = 10,dim = 200)


# 在实际的应用中 业务逻辑上 如果不是我的数据集处理的范围之内 先做个判断 就直接返回（不在我的业务范围内）了 减少模型的算例消耗 ->这就是调用大模型之前的意图识别


