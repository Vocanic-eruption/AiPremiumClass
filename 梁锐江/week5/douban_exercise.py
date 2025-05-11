"""
    BM25
    k的作用是控制词频的饱和度
        当k1较小时(如接近0):
            词频的贡献会迅速饱和，即使词频很高，对相关性评分的提升也会很快趋于平稳。
        当k2较大时:
            词频的贡献会更接近线性增长，高频词对相关性评分的影响会更大。
        对于低频词，BM25 希望它们能对相关性评分产生更大的影响，因为低频词通常更能反映文档的主题。
        对于高频词，BM25 希望它们的影响逐渐趋于饱和，避免过高的权重。
    b参数决定了文档长度归一化的影响
        当b = 0时，完全忽略文档长度的影响
        当b = 1时，完全按照文档长度的比例进行归一化
        一般情况下，b设置为0.75，表示适当地考虑文档长度的影响
"""

from nltk.corpus import stopwords
import csv
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy
import bm25_code

'''
加载评论:不分词的话,TfidfVectorizer只会分割标点符号
'''


def load_book_comments():
    book_comments_dict = {}
    with (open("./comments_fixed.txt", 'r', encoding='utf-8') as comment_file):
        reader = csv.DictReader(comment_file, delimiter='\t')
        for item in reader:
            book = item['book']
            body = item['body']

            if book == '' or body is None: continue

            # 默认返回的空列表没有与字典绑定 因此需要从字典中重新提取引用
            word_lines = jieba.lcut(body)
            book_comments_dict[book] = book_comments_dict.get(book, [])
            book_comments_dict[book].extend(word_lines)
    return book_comments_dict


def tfidf_fun(book_comments):
    vectorizer = TfidfVectorizer(stop_words=stop_words_cn)
    tfidf_matrix = vectorizer.fit_transform([' '.join(value) for value in book_comments.values()])
    # print(tfidf_matrix)
    # print(tfidf_matrix.shape)
    return cosine_similarity(tfidf_matrix)


def bm25_fun():
    bm25_similarity_matrix = bm25_code.bm25(book_comments.values(), k=15, b=0.75)
    # bm25_similarity_matrix = bm25_code.bm25([' '.join(value) for value in book_comments.values()], k=15, b=0.75)
    # print(bm25_similarity_matrix.shape)
    bm25_similarity = cosine_similarity(bm25_similarity_matrix)
    # print(bm25_similarity)
    # print(bm25_similarity.shape)
    return bm25_similarity


if __name__ == '__main__':
    stop_words_cn = stopwords.words('chinese')
    book_comments = load_book_comments()
    print(f'书籍长度:{len(book_comments)}')

    similarity_matrix = numpy.zeros((len(book_comments), len(book_comments)))

    book_names = list(book_comments.keys())
    print(book_names)

    while True:
        use_input_book_name = input("请输入图书名称:")
        operate_type = input("请选择推荐方式1-tfidf 2-bm25:")

        u_index = book_names.index(use_input_book_name)
        if operate_type == '1':
            similarity_matrix = tfidf_fun(book_comments)
        elif operate_type == '2':
            similarity_matrix = bm25_fun()
        # 取前三本
        recommend_book_indexs = numpy.argsort(-similarity_matrix[u_index])[1:4]
        for index in recommend_book_indexs:
            print(f"《{book_names[index]}》 \t 相似度: {similarity_matrix[u_index][index]:.4f}")
