
import csv
import jieba
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bm25_code import bm25


# 读取评论数据
def load_comments(file_path):
    comments = {} # 用于存储每本书的评论 # {书名: “评论1词 + 评论2词 + ...”}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')   # 识别格式文本中标题列
        for row in reader:
            book_name = row['book']
            comment_body = row['body']
            # 如果 book_name 或 comment_body 为空 或 None，则跳过该行
            if book_name is None or comment_body is None or book_name == "" or comment_body == "" :
                continue
            comment_words = jieba.lcut(comment_body)
             # 从comments中获取book_name的评论列表，如果列表不存在，则创建一个空列表
            comments[book_name] = comments.get(book_name,list())
            # 将comment_words添加到comments[book_name]中
            comments[book_name].extend(comment_words)
            # print(comments[book_name])
    return comments

# 分别构建TF-IDF，bm25矩阵，并计算相似度
def calculate_similarity(comments,method='tfidf'):
    if method == 'bm25':
        # 计算 BM25 矩阵
        matrix = bm25(comments, k=1.5, b=0.75)
    if method == 'tfidf':
        # 计算 TF-IDF 矩阵
        stopwords = [line.strip() for line in open('stopwords.txt', 'r', encoding='utf-8')]
        vectorizer = TfidfVectorizer(stop_words=stopwords)
        matrix = vectorizer.fit_transform([' '.join(comms) for comms in comments])

    # 计算相似度矩阵
    similarity_matrix = cosine_similarity(matrix)
    return similarity_matrix

if __name__ == '__main__':
    # 读取数据
    comments = load_comments('doubanbook_top250_comments_fixed.txt')
    book_list = list(comments.keys())
    book_comments = list(comments.values())  # 获取所有评论词列表，是一个二维列表
    #book_names = []  # 用于存储书名列表
    #book_comments = []
    #for book_name, comment in comments.items():
        # 将列表转换为字符串
        # book_names.append(book_name)
        # 将评论列表转换为字符串
        #book_comments.append(comment)

    # 相似度矩阵
    tfidf_matrix = calculate_similarity(book_comments, method='tfidf')
    bm25_matrix = calculate_similarity(book_comments, method='bm25')

    print(book_list)
    choose_input = input("请输入书名：")
    print("您输入的书名是：", choose_input)
    # 查找书名在列表中的索引
    book_index = book_list.index(choose_input)
    # tfidf相似度的前10个书名
    tfidf_similarities = tfidf_matrix[book_index]
    tfidf_similarities_indices = np.argsort(tfidf_similarities)[::-1][1:11]
    for index in tfidf_similarities_indices:
        print(f"TF-IDF相似度第{index+1}本书：{book_list[index]}，相似度：{tfidf_similarities[index]}")

    # bm25相似度的前10个书名
    bm25_similarities = bm25_matrix[book_index]
    bm25_similarities_indices = np.argsort(bm25_similarities)[::-1][1:11]
    for index in bm25_similarities_indices:
        print(f"BM25相似度第{index+1}本书：{book_list[index]}，相似度：{bm25_similarities[index]}")


    

   