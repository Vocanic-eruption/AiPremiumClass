#使用自定义的文档文本，通过fasttext训练word2vec训练词向量模型，并计算词汇间的相关度。（选做：尝试tensorboard绘制词向量可视化图）

import fasttext
from torch.utils.tensorboard import SummaryWriter

#处理文本，围城
model = fasttext.train_unsupervised('weicheng.txt', epoch=40) 
print('方鸿渐 近似词：', model.get_nearest_neighbors('方鸿渐'))