import fasttext
from fasttext import FastText

# model = fasttext.train_supervised('./cooking.stackexchange.txt')

# 检索单词和标签列表
# print(model.words)
# print(model.labels)

# 保存模型
# model.save_model('model_cooking.bin')

# 加载模型
model = FastText.load_model('model_cooking.bin')

# 返回预测概率最高的标签
model.predict('Which baking dish us best to bake a banana bread?')

# 通过指定k参数返回概率最高的k个标签
model.predict('Which baking dish us best to bake a banana bread?', k=3)

# 预测字符串数组
model.predict(['Which baking dish us best to bake a banana bread?', 'Why not put knives in the dishwasher?'], k=3)

