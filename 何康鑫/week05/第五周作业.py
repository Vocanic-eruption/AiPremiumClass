import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.summarization import bm25
import jieba

# 加载数据
df = pd.read_csv("douban_reviews.csv")
texts = df["comment"].tolist()

stop_words = set(open("stopwords.txt", "r", encoding="utf-8").read().splitlines())
processed_texts = []
for text in texts:
    words = [word for word in jieba.cut(text) if word not in stop_words]
    processed_texts.append(" ".join(words))

# TF-IDF实现
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_texts)
cos_sim = cosine_similarity(tfidf_matrix)

# BM25实现
bm25_model = bm25.BM25(processed_texts)
average_idf = sum(bm25_model.idf.values()) / len(bm25_model.idf)

# 推荐函数（以某条评论为例）
def recommend(query, model, data, top_n=5):
    query = " ".join([word for word in jieba.cut(query) if word not in stop_words])
    if isinstance(model, TfidfVectorizer):
        query_vec = model.transform([query])
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    elif isinstance(model, bm25.BM25):
        scores = model.get_scores(query)
        sim_scores = scores
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    return df.iloc[top_indices]

# 推荐与某条评论最相似的5条
print(recommend("这本书非常精彩！", tfidf_matrix, df))
print(recommend("这本书非常精彩！", bm25_model, df, average_idf))

import fasttext
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 训练词向量模型（skip-gram模式）
model = fasttext.train_unsupervised("my_documents.txt", model="skipgram")

# 计算词汇相似度
similarity = model.get_word_vector("apple").dot(model.get_word_vector("banana"))
print(f"Apple vs Banana Similarity: {similarity:.4f}")

# 1. 导出词向量到txt文件
model.save_vectors("vectors.txt")

# 2. 转换为TensorFlow格式
!tensorboard dev upload --logdir ./ --name "My Word Vectors"

import fasttext

# 训练分类模型
model = fasttext.train_supervised(
    input="cooking.stackexchange.txt",
    lr=0.5,  # 学习率
    dim=100, # 词向量维度
    epoch=25 # 训练轮次
)

# 测试模型
result = model.test("test.txt")
print(f"Precision@1: {result[1]:.3f}")

