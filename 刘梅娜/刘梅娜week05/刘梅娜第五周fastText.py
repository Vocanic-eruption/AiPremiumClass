import fasttext
import numpy as np
from tensorboardX import SummaryWriter
import torch

# 示例数据
with open('custom_text.txt', 'w') as f:
    f.write("This is a sample text for training word vectors.\n")
    f.write("Another example sentence to train the model.\n")
    f.write("Science fiction is a genre of speculative fiction.\n")
    f.write("The plot of a novel is a narrative of events, real or imagined.\n")
    f.write("A great book can transport you to another world.\n")

# 训练word2vec模型，调整minCount参数
model = fasttext.train_unsupervised('custom_text.txt', model='skipgram', minCount=1)

# 计算词汇间的相关度
word1 = 'sample'
word2 = 'example'
similarity = model.get_word_vector(word1).dot(model.get_word_vector(word2)) / (np.linalg.norm(model.get_word_vector(word1)) * np.linalg.norm(model.get_word_vector(word2)))
print(f"Similarity between '{word1}' and '{word2}': {similarity}")

# 可视化词向量（使用tensorboard）
writer = SummaryWriter()
word_vectors = torch.tensor([model.get_word_vector(word) for word in model.get_words()])
writer.add_embedding(word_vectors, metadata=model.get_words())
writer.close()