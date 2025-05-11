import fasttext

model = fasttext.train_supervised('cooking.stackexchange.txt', epoch=20, dim=300)

# word2vec模型使用方法，文本分类中一样可用

# 文本分类功能
print(model.predict("Which baking dish is best to bake a banana bread ?"))
print(model.predict("Baking chicken in oven, but keeping it moist"))