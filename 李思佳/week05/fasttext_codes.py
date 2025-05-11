import fasttext

model = fasttext.train_supervised('李思佳/week05/cooking.stackexchange.txt', epoch=10, dim=200)

#word2vec模型使用方法，文本分类中一样可用

# 文本分类功能
print(model.predict("What can I use instead of corn syrup?"))
print(model.predict("American equivalent for British chocolate terms"))