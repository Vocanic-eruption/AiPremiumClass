import csv
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import bm25_code # 引入bm25算法

import os
os.chdir(os.path.dirname(__file__))  # 切换到 .py 文件所在目录

#book	id	star	time	likenum	body
def load_books(file_path):
    book_comments = {} # 存储书籍评论的字典 {书名：评论1 + 评论2 + ...}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            book = row['book']
            comment = row['body']
            comment_words = jieba.cut(comment)
            if book == '': continue
            
            book_comments[book] = book_comments.get(book, []) 
            book_comments[book].extend(comment_words) # 将评论分词后添加到对应书籍的评论列表中
    return book_comments

if __name__ == '__main__':
    
    stop_words = [words for words in open('stopwords.txt', 'r', encoding='utf-8')]
    
    file_path = 'doubanbook_top250_comments_fixed.txt'  # 替换为你的文件路径
    book_comments = load_books(file_path)
    print(len(book_comments))

    book_names = []
    comments = []
    for book, comment in book_comments.items():
        book_names.append(book)
        comments.append(comment)  # 将评论列表转换为字符串
        
    # # 使用TfidfVectorizer计算TF-IDF矩阵
    # vectorizer = TfidfVectorizer(stop_words=stop_words)
    # tfidf_matrix = vectorizer.fit_transform([' '.join(comment) for comment in comments])
    # 计算余弦相似度
    # cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # 使用bm25算法计算相似度
    k = 1.5
    b = 0.75
    bm25_matrix = bm25_code.bm25(comments, k, b)  # 计算bm25矩阵
    # 计算余弦相似度
    cosine_similarities = cosine_similarity(bm25_matrix, bm25_matrix)
    
   
    # 推荐系统

    book_list = list(book_comments.keys())
    # print(book_list)
    book_name = input("请输入书名：")
    book_index = book_list.index(book_name)  
    
    recommendations = np.argsort(-cosine_similarities[book_index])[1:11]  # 排序并取前10个相似书籍的索引
    
    print("推荐书籍：")
    for i in recommendations:
        print(len(cosine_similarities[book_index]))
        print(f"<<{book_names[i]}>> \t相似度：{cosine_similarities[book_index][i]:.4f}")
        