import csv
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from data.bm25_code import bm25

# 加载图书评论信息
def load_book_comments(file_path):
    book_comments = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            book = row['book']
            comment = row['body']
            comment_word = jieba.lcut(comment)
            book_comments[book] = book_comments.get(book, [])
            book_comments[book].extend(comment_word)
    return book_comments

if __name__ == '__main__':
    # 加载停用词
    stopwords = [line.strip() for line in open('./data/raw/stopwords.txt', 'r', encoding='utf-8')]
    # 加载图书评论信息
    book_comments = load_book_comments('./data/fixed/doubanbook_top250_comments_fixed.txt')
    print('图书数量：', len(book_comments))
    # 获取书名和评论
    book_names_list = []
    book_comments_list = []
    book_comments_list_pure = [] # 去掉停用词
    for book, comments in book_comments.items():
        book_names_list.append(book)
        book_comments_list.append(comments)
        pure_comments = [item for item in comments if item not in stopwords]
        book_comments_list_pure.append(pure_comments)

    # 构建TF-IDF特征矩阵
    tfidfVectorizer = TfidfVectorizer(stop_words=stopwords)
    tfidf_matrix = tfidfVectorizer.fit_transform([' '.join(comments) for comments in book_comments_list])
    tfidf_similarity_matrix = cosine_similarity(tfidf_matrix) # 计算图书之间的余弦相似度

    # 构建bm25特征矩阵
    bm25_matrix = bm25(book_comments_list_pure)
    bm25_similarity_matrix = cosine_similarity(bm25_matrix) # 计算图书之间的余弦相似度

    # 根据输入的图书进行推荐
    print(book_names_list)
    book_name = input('请输入图书名称：')
    book_idx = book_names_list.index(book_name)
    print('====================基于TF-IDF算法的推荐=======================')
    tfidf_recommend_idx = np.argsort(-tfidf_similarity_matrix[book_idx])[1:11]
    for idx in tfidf_recommend_idx:
        print(f'《{book_names_list[idx]}》\t相似度：{tfidf_similarity_matrix[book_idx][idx]}')

    print('====================基于BM25算法的推荐=======================')
    bm25_recommend_idx = np.argsort(-bm25_similarity_matrix[book_idx])[1:11]
    for idx in bm25_recommend_idx:
        print(f'《{book_names_list[idx]}》\t相似度：{bm25_similarity_matrix[book_idx][idx]}')


