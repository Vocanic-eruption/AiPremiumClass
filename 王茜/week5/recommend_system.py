import csv
import jieba
from typing import DefaultDict

import numpy as np
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# bm25算法实现，输入为评论列表集合，k,b为超参数。输出所有评论的bm25结果矩阵
# 输入：comments = [['a','b','c'],['a','b','d'],['a','b','e']]
# 其中bm25[0] = [0.0, 0.0, 0.0, 0.0, 0.0]表示第一个评论的bm25值
# 其中bm25[0][0] = 0.0表示a的bm25值为0.0
def bm25(comments, stop_words= [], k=1.5, b=0.75):
    # 计算文档总数
    N = len(comments)
    # 初始化文档长度列表和词频字典
    doc_lengths = []
    word_doc_freq = {}
    doc_term_dict = [{} for _ in range(N)]

    for i, comment in enumerate(comments):
        # 记录文档长度
        # comment = list(set(comment).difference(stop_words))
        doc_lengths.append(len(comment))
        unique_words = set()
        for word in comment:
            # 统计词频
            doc_term_dict[i][word] = doc_term_dict[i].get(word, 0) + 1
            unique_words.add(word)
        # 统计包含该词的文档数量
        for word in unique_words:
            word_doc_freq[word] = word_doc_freq.get(word, 0) + 1

    # 计算每个单词的平均文档长度
    avg_doc_len = sum(doc_lengths) / N

    # 构建词汇表
    vocabulary = list(word_doc_freq.keys())
    word_index = {word: idx for idx, word in enumerate(vocabulary)}

    # 构建文档 - 词频矩阵
    doc_term_matrix = np.zeros((N, len(vocabulary)))
    for i in range(N):
        for word, freq in doc_term_dict[i].items():
            idx = word_index.get(word)
            if idx is not None:
                doc_term_matrix[i, idx] = freq

    # 计算 idf 值
    idf_numerator = N - np.array([word_doc_freq[word] for word in vocabulary]) + 0.5
    idf_denominator = np.array([word_doc_freq[word] for word in vocabulary]) + 0.5
    idf = np.log(idf_numerator / idf_denominator)
    idf[idf_numerator <= 0] = 0  # 避免出现 nan 值

    # 计算 bm25 值
    doc_lengths = np.array(doc_lengths)
    bm25_matrix = np.zeros((N, len(vocabulary)))
    for i in range(N):
        tf = doc_term_matrix[i]
        bm25 = idf * (tf * (k + 1)) / (tf + k * (1 - b + b * doc_lengths[i] / avg_doc_len))
        bm25_matrix[i] = bm25

    # 根据原始评论顺序重新排列 bm25 值
    final_bm25_matrix = []
    for i, comment in enumerate(comments):
        bm25_comment = []
        for word in comment:
            idx = word_index.get(word)
            if idx is not None:
                bm25_comment.append(bm25_matrix[i, idx])
        final_bm25_matrix.append(bm25_comment)

    # 找到最长的子列表长度
    max_length = max(len(row) for row in final_bm25_matrix)
    # 填充所有子列表到相同的长度
    padded_matrix = [row + [0] * (max_length - len(row)) for row in final_bm25_matrix]
    # 转换为 numpy 数组
    final_bm25_matrix = np.array(padded_matrix)
    return final_bm25_matrix


def load_comments(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        comments = {}
        for item in reader:
            book = item['book']
            comment = item['body']
            if comment is None:
                continue
            comments[book] = comments.get(book, []) + jieba.lcut(comment)

    book_list = []
    comment_list = []
    for k, v in comments.items():
        book_list.append(k)
        comment_list.append(v)
    return book_list, comment_list


def recommend_system(comment_list, stopwords, method='tfidf'):
    if method not in ['tfidf', 'bm25']:
        print(f'不存在 {method} 方法, 使用默认方法: tfidf')
    if method == 'tfidf':
    # 构建TF-IDF  矩阵
        tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)
        matrix = tfidf_vectorizer.fit_transform(comment_list)
    if method == 'bm25':
        matrix = bm25(comment_list, stopwords, k=1.5, b=0.75)
    # 余弦相似度计算
    similarity = cosine_similarity(matrix)
    print(similarity.shape)
    return similarity


def get_similar_books(book_list, similarity):
    recommend_books = {}
    for i, book in enumerate(book_list):
        ss = similarity[i, :]
        book_similarity = -ss
        recommend_books_idxes = np.argsort(book_similarity)[:11][1:]
        recommend_books[book] = [(book_list[i], ss[i]) for i in recommend_books_idxes]
    return recommend_books


def main():
    return

if __name__ == '__main__':
    douban_path = 'doubanbook_fixed_comments.txt'
    stopwords_path = 'stopwords.txt'
    stopwords = [x.strip() for x in open(stopwords_path, 'r')]
    main()
    books, comments = load_comments(douban_path)
    print(len(comments))
    tfidf_comments = [' '.join(x) for x in comments]
    tfidf_similarity = recommend_system(tfidf_comments, stopwords, method='tfidf')
    bm25_similarity = recommend_system(comments, stopwords, method='bm25')
    books_name = "\n".join(books)
    tfidf_recommend_books = get_similar_books(books, tfidf_similarity)
    bm25_recommend_books = get_similar_books(books, bm25_similarity)
    print(f'可选书名为：\n{books_name}')
    while True:
        book = input(f'请输入书名：如需退出，请直接中断程序\n')
        if book not in books:
            print(f'无此书名，重新输入\n')
            break
        tfidf_book = tfidf_recommend_books[book]
        bm25_book = bm25_recommend_books[book]
        print(f'输入书： {book}')
        print('tfidf 推荐')
        for book, similarity in tfidf_book:
            print(f'book: {book}, similarity: {similarity}')
        print('bm25 推荐')
        for book, similarity in bm25_book:
            print(f'book: {book}, similarity: {similarity}')
