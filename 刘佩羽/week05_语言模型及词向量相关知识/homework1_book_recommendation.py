import csv
import pickle
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bm25_code import bm25


def load_data(file_path):
    # 图书评论信息集合
    books_comments = {}  # {书名: "评论1,+评论2+ ..."}
    # 打开文件，以只读模式读取，编码格式为utf-8
    with open(file_path, 'r', encoding='utf-8') as f:
        # 使用csv.DictReader读取文件，以制表符作为分隔符
        reader = csv.DictReader(f, delimiter='\t')
        # 遍历读取的每一行
        for item in reader:
            # print(item['book'], item['body'])
            # 获取书籍信息
            book = item['book']
            # 获取评论内容
            comment = item['body']
            # 使用jieba分词对评论内容进行分词
            comment_words = jieba.lcut(comment)
            # 图书评论收集
            books_comments[book] = books_comments.get(book, [])  # 如果书名不存在，则创建一个空列表
            books_comments[book].extend(comment_words)  # 将评论中的词语添加到书名对应的列表中
    return books_comments


# 如果当前模块是主模块，则执行以下代码
if __name__ == '__main__':

    # 加载停用词
    stop_words = [line.strip() for line in open('./stopwords.txt', 'r', encoding='utf-8').readlines()]

    # # 加载图书评论数据
    # books_comments = load_data('./doubanbook_top250_comments_fixed.txt')
    # print('书籍数量', len(books_comments))
    # # 写一段代码缓存结果，避免重复计算
    # with open('./books_comments.pkl', 'wb') as f:
    #     pickle.dump(books_comments, f)

    # 加载缓存
    with open('./books_comments.pkl', 'rb') as f:
        books_comments = pickle.load(f)
    # 创建一个空列表，用于存储书籍名称
    book_names = []
    # 创建一个空列表，用于存储书籍评论
    books_comments_list = []
    # 遍历books_comments字典中的每个元素
    for book, comment in books_comments.items():
        # 将书籍名称添加到book_names列表中
        book_names.append(book)

        # 将书籍评论添加到books_comments_list列表中，评论以空格连接成一个字符串
        books_comments_list.append(' '.join(comment))

    # 构建TF-IDF特征矩阵
    tfidf = TfidfVectorizer(stop_words=stop_words)      # 创建一个TF-IDF向量化器，并设置停用词
    tfidf_matrix = tfidf.fit_transform(books_comments_list)     # 将书籍评论转换为TF-IDF矩阵
    
    print('TF-IDF算法')
    # 打印TF-IDF矩阵的形状
    print('TF-IDF矩阵的形状', tfidf_matrix.shape)

    # 计算图书之间的相似度
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)  # 计算tfidf矩阵的余弦相似度矩阵
    print('相似度矩阵的形状', similarity_matrix.shape)  # 打印相似度矩阵的形状

    # print(similarity_matrix[:5, :5])  # 打印相似度矩阵的前5行和前5列
    # 输入要推荐相似的图书的名称
    book_name = input('请输入要推荐相似的图书的名称：')
    # book_name = '盗墓笔记'

    # 找到输入的图书在book_names列表中的索引
    book_index = book_names.index(book_name)

    # 获取与当前书籍相似度最高的十个书籍的索引
    top_10_indices = np.argsort(-similarity_matrix[book_index])[1:5]

    # 输出与输入的图书最相似的图书的名称和相似度
    for index in top_10_indices:
        print('相似图书：', book_names[index],
              '相似度：', similarity_matrix[book_index][index]*100, '%')
    # 输出与输入的图书最相似的图书的评论
    # print('评论：', books_comments[book_names[most_similar_book_index]])

    print('BM25算法')
    # bm25特征矩阵
    tfidf_matrix = bm25(books_comments_list)
    # 计算图书之间的相似度
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)  # 计算tfidf矩阵的余弦相似度矩阵
    print('相似度矩阵的形状', similarity_matrix.shape)  # 打印相似度矩阵的形状

    # 获取与当前书籍相似度最高的十个书籍的索引
    top_10_indices = np.argsort(-similarity_matrix[book_index])[1:5]

    # 输出与输入的图书最相似的图书的名称和相似度
    for index in top_10_indices:
        print('相似图书：', book_names[index],
              '相似度：', similarity_matrix[book_index][index]*100, '%')
