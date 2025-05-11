import fasttext

file_name = 'cooking.stackexchange.txt'

# 训练模型
model = fasttext.train_supervised(file_name, label_prefix='__label__', epoch=100, lr=0.1)

# 检索单词和标签列表
words_list = model.get_words()
# print(words_list)

labels_list = model.get_labels()
# print(labels_list)

# 预测标签
sentence1 = 'Basic carrier sauce/syrup for different sweet flavors?'
label1 = model.predict(sentence1)
print(label1)

sentence2 = 'What kind of tea do you boil for 45minutes?'
label2 = model.predict(sentence2)
print(label2)
