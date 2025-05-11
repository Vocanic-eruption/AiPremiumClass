import fasttext

# 训练模型
model = fasttext.train_supervised(input="C:/Users/stephanie.chen/miniconda3/envs/py312/cooking.stackexchange.txt", epoch=25, lr=1.0, wordNgrams=2, verbose=2, minCount=1, loss='softmax')

# 保存模型
model.save_model("model_cooking.ftz")

# 加载模型
model = fasttext.load_model("model_cooking.ftz")

# 预测示例
text = "How to make mayonnaise without eggs?"
labels, probabilities = model.predict(text, k=3)  # top 3 标签
print(labels, probabilities)