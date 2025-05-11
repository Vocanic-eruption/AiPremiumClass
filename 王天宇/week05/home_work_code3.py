#使用课堂示例cooking.stackexchange.txt，使用fasttext训练文本分类模型。（选做：尝试使用Kaggle中的Fake News数据集训练文本分类模型）
import fasttext

model = fasttext.train_supervised('cooking.txt', epoch=20, dim=100)

print(model.predict('Can you boil the potatoes for mashed potatoes too long?'))
#(('__label__potatoes',), array([0.86901009]))
