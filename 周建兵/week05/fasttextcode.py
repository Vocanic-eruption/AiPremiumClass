import fasttext
import os
os.chdir(os.path.dirname(__file__))

model = fasttext.train_supervised("cooking.stackexchange.txt")

# 文本分类
print(model.predict("I love cooking!"))
print(model.predict("I hate cooking!"))

