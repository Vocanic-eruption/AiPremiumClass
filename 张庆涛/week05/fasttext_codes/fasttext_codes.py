import fasttext
import os
path ='cooking.stackexchange_model.bin'
if os.path.exists(path):
  model= fasttext.load_model('cooking.stackexchange_model.bin')
else:
   model = fasttext.train_supervised('cooking.stackexchange.txt',epoch=200,dim=400,lr=0.9,wordNgrams=8)
   model.save_model(path)

# word2vec 模型使用该方法，文本分类中一样可用



# 文本分类功能
# 
# 
print(model.predict(['How to know whether the oven\'s door has loosened and leaking out some energy or is as tight as it was when new?','Can you make Bearnaise with olive oil?','What is the easiest way to remove chicken leg/drumstick tendons?']))