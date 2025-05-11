import torch.nn as nn
import torch
import torch.optim as optim
import kagglehub
import os
import pandas as pd
import numpy as np
import jieba
from sklearn.model_selection import train_test_split

# Download latest version
# path = kagglehub.dataset_download("utmhikari/doubanmovieshortcomments")
path = 'C:/Users/ligt/.cache/kagglehub/datasets/utmhikari/doubanmovieshortcomments/versions/7/'
print("Path to dataset files:", path)

data_path = os.path.join(path,"DMSC.csv")

data = pd.read_csv(data_path)
data = data[['Star','Comment']]

#star==1,2是正例，4，5是负例，定义转换接口
def convert_star(Star):
    if Star in [1,2]:
        return 1
    elif Star in [4,5]:
        return 0
    else:
        return -1
data['Star'] = data['Star'].apply(convert_star)

# 删除了转换为 -1的行，包括评论一起去除掉
data = data[data['Star'] != -1]

#加载停用词
stop_set = set()
def load_stop():
    with open('C:/Users/ligt/bd_AI/w07/stop_words.txt','r',encoding='utf-8') as f:
        for line in f:
            stop_set.add(line.strip())
            
    return stop_set

stop_w = load_stop()

#构建词汇表
#isse修改：vocal.add(words) 应该改为 vocal.update(words)，因为 words 是一个列表，而 add 方法只接受单个元素
vocal = set()
for comment in data['Comment']:
    comment = jieba.lcut(comment)
    words = [word for word in comment if word not in stop_w] #去掉停用词
    vocal.update(words)
    
vocal = sorted(vocal)  #按照字典顺序排序
vocal_size = len(vocal)
print(f"词汇表大小：{len(vocal)}")

# # 创建词到索引的映射，将文本转换为索引序列 0:a 1:b 
word2idx = {word:i for i ,word in enumerate(vocal)}

# 将文本转换为索引序列
text_idx = []
for ti in data['Comment']:
    t2i = jieba.lcut(ti)
    t2i_s = [ws for ws in t2i if ws not in stop_w]
    idx_seq = [word2idx[idx] for idx in t2i_s if idx in word2idx]  #将评论先分词->去停用词->根据序列表将分词转成索引序列的id然后保存
    text_idx.append(torch.tensor(idx_seq))