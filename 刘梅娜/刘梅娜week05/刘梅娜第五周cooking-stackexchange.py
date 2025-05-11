import fasttext

# 假设我们有一个文本分类数据集 'cooking.stackexchange.txt'

# 示例数据
with open('cooking.stackexchange.txt', 'w') as f:
    f.write("__label__cooking How do I make a cake?\n")
    f.write("__label__cooking What is the best way to bake bread?\n")
    f.write("__label__recipes What are some easy recipes for beginners?\n")

# 训练文本分类模型
model = fasttext.train_supervised('cooking.stackexchange.txt')

# 预测示例
text = "How can I make pizza?"
prediction = model.predict(text)
print(f"Predicted label: {prediction[0][0]}, Confidence: {prediction[1][0]}")

# 使用Kaggle中的Fake News数据集训练文本分类模型
# 下载数据集并预处理
# model = fastText.train_supervised('fake_news_train.txt')