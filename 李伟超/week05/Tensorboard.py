from fasttext import FastText
import torch
from torch .utils.tensorboard import SummaryWriter


wv_file = 'hlm_cut.txt' # 训练语料文件路径。

# 训练fasttext词向量模型。
model = FastText.train_unsupervised(wv_file, epoch=10, dim=200) # 训练模型，生成词向量。
model.save_model('model.bin') # 保存模型。

# 加载预训练的fasttext词向量
word_vectors = FastText.load_model(wv_file) # 加载预训练的fasttext词向量。

# 获取词汇表大小和词向量维度
vocab_size = len(word_vectors.words) # 获取词汇表大小。
embedding_dim = word_vectors.get_dimension() # 获取词向量维度。
print(vocab_size)
print(embedding_dim)

# 创建一个嵌入矩阵，每一行都是一个词的Fastetxt词向量。
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for i,word in enumerate(word_vectors.words): # 遍历词汇表中的每个词。
    embedding_vector = word_vectors(word) 
    if embedding_vector is not None: # 如果词向量存在。
        embedding_matrix[i] = embedding_vector # 将词向量赋值给嵌入矩阵的对应行
        
embedding = nn.Embedding.from_pretrained(torch.fLoatTensor(embedding_matrix))


writer = SummaryWriter()
meta= []
while len(meta) < 100:
    i = len(meta)
    meta = meta + word_vectors.words[i]
meta = meta[:100]

writer.add_embedding(embedding.weight[:100], metadata=meta)