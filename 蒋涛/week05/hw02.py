"""
2. 使用自定义的文档文本，通过fasttext训练word2vec训练词向量模型，并计算词汇间的相关度。（选做：尝试tensorboard绘制词向量可视化图）
"""

import fasttext
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

# 自定义文档文本
documents = [
    "我喜欢阅读科幻小说",
    "科幻小说能开拓我的思维",
    "我也喜欢看悬疑电影",
    "悬疑电影充满了惊喜和挑战"
]

# 保存文档到文件
with open('custom_text.txt', 'w', encoding='utf-8') as f:
    for doc in documents:
        f.write(doc + '\n')

# 训练fasttext词向量模型，降低minCount的值
model = fasttext.train_unsupervised('custom_text.txt', model='skipgram', minCount=1)

# 计算词汇间的相关度
word1 = "科幻小说"
word2 = "悬疑电影"
similarity = model.get_word_vector(word1).dot(model.get_word_vector(word2)) / (
        np.linalg.norm(model.get_word_vector(word1)) * np.linalg.norm(model.get_word_vector(word2)))
print(f"{word1} 和 {word2} 的相关度为: {similarity}")

# 选做：使用tensorboard绘制词向量可视化图
def visualize_word_vectors(model, log_dir='./logs'):
    writer = SummaryWriter(log_dir)
    words = model.get_words()
    vectors = [model.get_word_vector(word) for word in words]
    writer.add_embedding(np.array(vectors), metadata=words)
    writer.close()
    print(f"词向量可视化数据已保存到 {log_dir}，可以使用tensorboard进行查看。")

# 调用函数生成可视化数据
visualize_word_vectors(model)

# 启动tensorboard
os.system(f"tensorboard --logdir=./logs")