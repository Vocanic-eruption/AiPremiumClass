import fasttext
import os

model = fasttext.train_supervised('./data/raw/cooking.stackexchange.txt')
# 创建文件夹
os.makedirs('./data/model', exist_ok=True)
model.save_model('./data/model/cooking_model.bin')