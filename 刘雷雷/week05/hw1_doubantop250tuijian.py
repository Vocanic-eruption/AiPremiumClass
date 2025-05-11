# 1.实现基于豆瓣top250图书评论的简单推荐系统（TF-IDF及BM25两种算法实现）
import csv
import jieba
import os
import numpy as np
from bm25 import bm25
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

current_dir = os.path.dirname(os.path.abspath(__file__))


# 加载数据
def load_data(file_name):
    # 图书评论信息集合
    book_comments = {}  # {书名: "评论词1 评论词2 ..."}

    with open(file_name, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for item in reader:
            # 书名
            book_name = item["book"]
            # 评论内容
            comment = item["body"]
            # 将评论内容分词
            comment_words = jieba.lcut(comment)

            # 跳过空书名
            if book_name == "":
                continue

            book_comments[book_name] = book_comments.get(book_name, [])
            book_comments[book_name].extend(comment_words)

    return book_comments


def comments_vectors_similarity(book_comments, method="bm25"):

    if method == "bm25":
        # 计算BM25特征矩阵
        # matrix = bm25([" ".join(comms) for comms in book_comments])
        matrix = bm25(book_comments)
    elif method == "tfidf":
        # 计算TF-IDF特征矩阵
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([" ".join(comms) for comms in book_comments])
    else:
        raise ValueError("method must be 'bm25' or 'tfidf'")

    # 计算余弦相似度
    similarity_matrix = cosine_similarity(matrix)
    return similarity_matrix


if __name__ == "__main__":
    # 加载停用词
    stop_words = [
        line.strip()
        for line in open(
            f"{current_dir}/stopwords.txt", "r", encoding="utf-8"
        ).readlines()
    ]
    # 加载数据
    book_comments_data = load_data(f"{current_dir}/doubanbook_comments_fixed.txt")

    # 提取书名和评论文本
    book_names = []
    book_comments = []
    for book_name, comments in book_comments_data.items():
        book_names.append(book_name)
        book_comments.append(comments)

    # 计算BM25算法得到相似度矩阵
    bm25_matrix = comments_vectors_similarity(book_comments, method="bm25")
    # 计算TF-IDF算法得到相似度矩阵
    tfidf_matrix = comments_vectors_similarity(book_comments, method="tfidf")

    # 输入要推荐的图书
    book_list = list(book_comments_data.keys())
    print(book_list)
    book_name = input("请输入要推荐的图书：")
    # 获取推书索引
    book_idx = book_names.index(book_name)

    print("BM25算法推荐：\n")
    recommend_book_index = np.argsort(-bm25_matrix[book_idx])[:11]
    for i in recommend_book_index:
        print(f"<<{book_names[i]}>> \t 相似度：{bm25_matrix[book_idx][i]}")

    print("\nTF-IDF算法推荐：\n")
    recommend_book_index = np.argsort(-tfidf_matrix[book_idx])[:11]
    for i in recommend_book_index:
        print(f"<<{book_names[i]}>> \t 相似度：{tfidf_matrix[book_idx][i]}")
