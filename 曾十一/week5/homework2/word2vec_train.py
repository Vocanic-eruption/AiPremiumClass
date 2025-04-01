##     2. 使用自定义的文档文本，通过fasttext训练word2vec训练词向量模型，并计算词汇间的相关度。（选做：尝试tensorboard绘制词向量可视化图）

import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def train_word2vec_model(input_file, output_model):
    
    # 训练模型
    model = fasttext.train_unsupervised(
        input=input_file,
        model='skipgram',  # 使用 SkipGram 模型
        dim=100,          # 词向量维度
        epoch=10,         # 训练轮数
        lr=0.05           # 学习率
    )
    
    # 保存模型
    model.save_model(output_model)
    print(f"模型已保存到 {output_model}")


def compute_word_similarity(model, word1, word2):
    
    vector1 = model.get_word_vector(word1)
    vector2 = model.get_word_vector(word2)
    
    # 将一维数组转换为二维数组
    vector1 = vector1.reshape(1, -1)
    vector2 = vector2.reshape(1, -1)
    similarity  = cosine_similarity(vector1, vector2)
    return similarity[0][0]



def main():
    # 输入文件路径
    input_file = "/mnt/data_1/zfy/4/week5/homework_2/jianlai_1.txt"
    # 输出模型路径
    output_model = "/mnt/data_1/zfy/4/week5/homework_2/jianlai_1.bin"
    # 训练词向量模型
    train_word2vec_model(input_file, output_model)
    # 加载模型
    model = fasttext.load_model(output_model)
    # 需要计算相似度的词汇列表
    words = ["陈平安", "宋集薪", "穷", "老师"]

    print(model.get_analogies("陈平安", "宋集薪", "穷"))
    print(model.get_nearest_neighbors("陈平安"))
    
    # 计算词汇间的相似度
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            similarity = compute_word_similarity(model, words[i], words[j])
            print(f"{words[i]} 和 {words[j]} 的相似度: {similarity:.4f}")


if __name__ == "__main__":
    main() 