import jieba
import os
import fasttext
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
hlm_path = os.path.join(current_dir, 'HLM.txt')
hlm_fixed_path = os.path.join(current_dir, 'HLM_fixed.txt')

def hlm_to_utf8(file_path,hlm_fixed_path):
    """
    将HLM.txt文件转换为utf-8编码格式，并进行分词处理，结果保存为HLM_fixed.txt
    """
    # 读取HLM.txt文件，GBK编码
    with open(file_path, 'r', encoding='GBK') as f:
        content = f.read()
        # 使用jieba进行分词
        contents = jieba.lcut(content)
        # 将分词结果写入到HLM_fixed.txt文件中
        with open(hlm_fixed_path, 'w', encoding='utf-8') as utf8_file:
            utf8_file.write(' '.join(contents))

# 使用fasttext 对 HLM_fixed.txt文件进行模型训练
def train_word2vec(file_path):
    # 无监督训练
    model_skipgram = fasttext.train_unsupervised(file_path, model='skipgram', lr=0.05, epoch=10, wordNgrams=2)
    model_cbow = fasttext.train_unsupervised(file_path, model='cbow', lr=0.05, epoch=10, wordNgrams=2)
    # 保存模型
    model_skipgram.save_model(os.path.join(current_dir,'fasttext_model_skipgram.bin'))
    model_cbow.save_model(os.path.join(current_dir,'fasttext_model_cbow.bin'))

# 使用fasttext加载模型
def load_model(model_path):
    model = fasttext.load_model(model_path)
    return model

# 获取词向量并计算相关度
def get_word_vector_and_similar_similarity(model, word1, word2):
    # 获取词向量
    word1_vector = model.get_word_vector(word1)
    word2_vector = model.get_word_vector(word2)
    # 计算余弦相似度
    word1_vector = word1_vector.reshape(1, -1)  # 转换为二维数组
    word2_vector = word2_vector.reshape(1, -1)  # 转换为二维数组
    # 计算余弦相似度
    similarity = cosine_similarity(word1_vector, word2_vector)
    return word1_vector, word2_vector, similarity
    
 # 获取词的最近邻
def get_nearest_neighbors(model, word):
    # 获取词向量
    word_vector = model.get_word_vector(word)   
    # 获取相似词
    similar_words = model.get_nearest_neighbors(word)
    return similar_words

# 获取句子的向量，并计算句子相似度
def get_sentence_vector_and_similarity(model, sentence1, sentence2):
    # 获取句子向量
    sentence1_vector = model.get_sentence_vector(sentence1)
    sentence2_vector = model.get_sentence_vector(sentence2)
    # 计算余弦相似度
    sentence1_vector = sentence1_vector.reshape(1, -1)  # 转换为二维数组
    sentence2_vector = sentence2_vector.reshape(1, -1)  # 转换为二维数组
    similarity = cosine_similarity(sentence1_vector, sentence2_vector)
    return sentence1_vector, sentence2_vector, similarity

# 获取句子的最近邻
def get_sentence_nearest_neighbors(model, sentence):
    # 获取句子向量
    sentence_vector = model.get_sentence_vector(sentence)   
    # 获取相似句子
    similar_sentences = model.get_nearest_neighbors(sentence)
    return similar_sentences

# tensorboard可视化
def visualize_embeddings(model, word_list):
    writer = SummaryWriter()
    word_vectors = []
    # 获取词向量
    for word in word_list:
        vector = model.get_word_vector(word)
        word_vectors.append(vector)
    # 添加到tensorboard
    # 先将列表转换为 numpy.ndarray
    word_vectors_np = np.array(word_vectors)
    writer.add_embedding(torch.tensor(word_vectors_np),word_list)
    # 关闭 SummaryWriter
    writer.close()

if __name__ == '__main__':
    # hlm_to_utf8(hlm_path,hlm_fixed_path)
    # train_word2vec(hlm_fixed_path)
    # 加载模型
    model_skipgram = load_model(os.path.join(current_dir,'fasttext_model_skipgram.bin'))
    
    # 获取词向量和相似度
    word1 = "贾宝玉"
    word2 = "林黛玉"
    word1_vector, word2_vector, similarity = get_word_vector_and_similar_similarity(model_skipgram, word1, word2)
    print(f"词1: {word1}, 词2: {word2}, 相似度: {similarity}")
    
    # 获取词的最近邻
    similar_words = get_nearest_neighbors(model_skipgram, word1)
    print(f"'{word1}'的最近邻:")
    for similar_word, similarity in similar_words:
        print(f"相似词: {similar_word}, 相似度: {similarity}")
    
    # 获取句子的向量和相似度
    sentence1 = "贾宝玉爱林黛玉"
    sentence2 = "林黛玉爱贾宝玉"
    sentence1_vector, sentence2_vector, similarity = get_sentence_vector_and_similarity(model_skipgram, sentence1, sentence2)
    print(f"句子1: {sentence1}, 句子2: {sentence2}, 相似度: {similarity}")
    
    # 获取句子的最近邻
    similar_sentences = get_sentence_nearest_neighbors(model_skipgram, sentence1)
    print(f"'{sentence1}'的最近邻:")
    for similar_sentence, similarity in similar_sentences:
        print(f"相似句子: {similar_sentence}, 相似度: {similarity}")

    # 可视化词向量
    word_list = ["贾宝玉", "林黛玉","袭人","葬花","大观园"]  # 可以添加更多的词
    visualize_embeddings(model_skipgram, word_list)
