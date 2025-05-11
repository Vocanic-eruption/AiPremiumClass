import numpy as np
import jieba
import csv

from sklearn.metrics.pairwise import cosine_similarity


def bm25(comments, k=1.5, b=0.75):
    # 计算文档的总数
    N = len(comments)
    # 初始化文档的长度列表和词频字典

    doc_lengths = []
    word_doc_freq = {}
    doc_term_dict = [{} for _ in range(N)]

    for idx, comment in enumerate(comments):
        doc_lengths.append(len(comment))
        unique_words = set()
        for word in comment:
            # 统计词频
            doc_term_dict[idx][word] = doc_term_dict[idx].get(word, 0) + 1
            unique_words.add(word)
        # 统计包含该词的文档数量
        for word in unique_words:
            word_doc_freq[word] = word_doc_freq.get(word, 0) + 1
    # 计算每个单词的平均文档长度
    avg_doc_len = sum(doc_lengths) / N

    # 构建词汇表
    vocabulary = list(word_doc_freq.keys())
    word_index = {word: idx for idx, word in enumerate(vocabulary)}

    # 构建文档-词频矩阵
    doc_term_matrix = np.zeros((N, len(vocabulary)))
    for i in range(N):
        for word, freq, in doc_term_dict[i].items():
            idx = word_index.get(word)
            if idx is not None:
                doc_term_matrix[i, idx] = freq

    # 计算idf值
    idf_numerator = N - np.array([word_doc_freq.get(word, 0) for word in vocabulary]) + 0.5
    idf_denominator = np.array([word_doc_freq.get(word, 0) for word in vocabulary]) + 0.5
    idf = np.log(idf_numerator / idf_denominator)
    idf[idf_numerator <= 0] = 0

    # 计算bm25的值
    doc_lengths = np.array(doc_lengths)
    bm25_matrix = np.zeros((N, len(vocabulary)))
    for i in range(N):
        tf = doc_term_matrix[i]
        bm25 = idf * (tf * (k + 1)) / (tf + k * (1 - b + b * doc_lengths[i] / avg_doc_len))
        bm25_matrix[i] = bm25

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
    # 填充所有子列表到相同长度
    padded_matrix = [row + [0] * (max_length - len(row)) for row in final_bm25_matrix]
    final_bm25_matrix = np.array(padded_matrix)
    return final_bm25_matrix


def load_data(file_name):
    book_comments = {}
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            book = row['book']
            comments = row['body']
            if book == '' or comments is None:
                continue
            comment_words = jieba.lcut(comments)
            # print(comment_words)
            book_comments[book] = book_comments.get(book, [])
            book_comments[book].extend(comment_words)
    return book_comments

if __name__ == '__main__':
    book_comments = load_data('./comments_fix.txt')
    books_name = []
    books_comment = []
    for book, comments in book_comments.items():
        books_name.append(book)
        books_comment.append(comments)
    bm25_matrix = bm25(comments=book_comments, k=1.5, b=0.75)
    print(bm25_matrix.shape)
    similarities_matrix = cosine_similarity(bm25_matrix)
    print(similarities_matrix.shape)
    book_list = list(book_comments.keys())
    print(book_list)
    book_name = input("请输入图书名称：")
    book_idx = books_name.index(book_name)
    recommend_book_index = np.argsort(-similarities_matrix[book_idx])[1:11]
    for idx in recommend_book_index:
        print(f"《{books_name[idx]}》 \t 相似度：{similarities_matrix[book_idx][idx]:.4f}")
