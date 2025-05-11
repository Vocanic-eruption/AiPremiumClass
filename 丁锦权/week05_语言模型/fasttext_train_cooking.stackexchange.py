import fasttext

model = fasttext.train_supervised('week5/cooking.stackexchange.txt')
# 文本分类功能
print(model.predict("Dangerous pathogens capable of growing in acidic environments"))
print(model.predict("Which plastic wrap is okay for oven use?"))