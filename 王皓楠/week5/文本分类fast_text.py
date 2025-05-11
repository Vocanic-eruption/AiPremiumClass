import fasttext
import numpy as np
from sklearn.manifold import TSNE
import torch
from torch.utils.tensorboard import SummaryWriter
import sys
import io
import jieba
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
#有监督学习，因为带了label和classname标签
model = fasttext.train_supervised('cooking.stackexchange.txt')
print(model.predict('What determines total heat when using chilis? Quantity  intensity?'))

