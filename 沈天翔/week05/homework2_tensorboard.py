from fasttext import FastText
import torch
from torch.utils.tensorboard import SummaryWriter

# 模型文件
model_file = 'san_guo.bin'
# 加载模型FastText词向量
word_vectors = FastText.load_model(model_file)

# print(word_vectors.words)
# word =  word_vectors.words[100]
# print(word) # 他
# print(word_vectors[word]) # 词向量
# print(word_vectors[word].shape) # (100,)

# 获取词汇表大小
vocab_size = len(word_vectors.words)
# print(vocab_size) # 3879

# 获取词向量维度
embedding_dim = word_vectors.get_dimension()
# print(embedding_dim) # 100

# 创建一个嵌入矩阵，每一行都是一个词的 FastText 词向量
embedding_matrix = torch.zeros((vocab_size, embedding_dim))
# print(embedding_matrix.shape) # torch.Size([3879, 100])

# 填充嵌入矩阵
for i, word in enumerate(word_vectors.words):
    embedding_vector = word_vectors[word]
    # print(embedding_vector)
    if embedding_vector is not None:
        embedding_matrix[i] = torch.FloatTensor(embedding_vector)

# print(embedding_matrix)

# 在模型中使用预训练的 FastText 词向量
emmbedding_layer = torch.nn.Embedding.from_pretrained(embedding_matrix)

# tensorboard 词向量可视化
writer = SummaryWriter()
meta = word_vectors.words[:100]

# print(meta)

writer.add_embedding(emmbedding_layer.weight[:100], metadata=meta)
writer.close()

