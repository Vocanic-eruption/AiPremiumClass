from fasttext import FastText


"""
wc cooking.stackexchange.txt    行：15404  单次：169582 字节：1401900

# 将前12404行作为训练集
head -n 12404 cooking.stackexchange.txt > cooking.train

# 将最后3000行作为验证集
tail -n 3000 cooking.stackexchange.txt > cooking.valid

"""
# 1、下载数据集并进行预处理，查看数据集的行数并将其分成训练集和验证集
# 2、用train data 来训练模型，通过fasttext中的train_supervised来训练模型
# 3、用valid data 来评估模型的效果，通过fasttext中的test来评估模型的效果
# 4、保存模型，用fasttext中的save_model来保存模型
# 5、加载模型，用fasttext中的load_model来加载模型
# 6、使用模型，用fasttext中的predict来使用模型

# 1、训练分类模型直接调用.train_supervised即可，必须传入input_path
model = FastText.train_supervised('cooking.train',lr=0.01,epoch=5)
# print('-------------------------------------')
# print(model.words[0])
# print(model.labels[0]) 
# 2、保存训练后的模型
model.save_model('cooking_model.bin')
# 3、可以使用valid data 来评估模型的效果，直接使用model.test即可,传入验证数据集
result = model.test('cooking.valid')
print(f'test result:{result}')
# 4、加载模型来查看
model = FastText.load_model('cooking_model.bin')
text = 'wihch is the best way to cook a steak'
pred = model.predict(text)
print(f'model pred:{pred}')