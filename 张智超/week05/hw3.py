import fasttext

# 加载模型
model = fasttext.load_model('./data/model/cooking_model.bin')

# 进行预测
label, pred = model.predict(['a bad odor'])
print(label)
print(pred)