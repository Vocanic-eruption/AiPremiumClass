import fasttext

# 训练模型
model = fasttext.train_supervised(input="cooking.stackexchange.txt", dim=50, epoch=25, lr=1.0, wordNgrams=2, loss="softmax")
# 训练好的模型保存到本地
model.save_model("cooking_model.bin")

'''word2vec里的方法，文本分类也可以使用'''

# 加载模型
model = fasttext.load_model("cooking_model.bin")
# 预测文本分类
print(model.predict("Pre-Cooking Steak in a bag, Any suggestions?水浒传.txt水浒传.txt"))