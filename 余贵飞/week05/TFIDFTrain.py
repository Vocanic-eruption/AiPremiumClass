# encoding: utf-8
# @File  : TFIDFTrain.py
# @Author: GUIFEI
# @Desc : 
# @Date  :  2025/03/27
import csv
import json

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
def load_data(filename):
    # 图书评论信息集合 # {书名： “评论1词 + 评 论2词 + ….”}
    book_comments = {}
    with open(filename,'r', encoding= 'utf-8') as f:
        reader = csv.DictReader(f,delimiter='\t') # 识别格式文本中标题列
        for item in reader:
            book = item['book']
            comment = item['body']
            if comment is not None:
                comment_words = jieba.lcut(comment)
                if book == '': continue # 跳过空书名
                # 图书评论集合收集
                book_comments[book] = book_comments.get(book,[])
                book_comments[book].extend(comment_words)
    # 保存分词之后的字典数据
    with open("../dataset/douban/boot_json.json",'w', encoding= 'utf-8') as f:
        json.dump(book_comments, f, indent=4, ensure_ascii=False)

def tfIdf(book_name, book_comments, stop_words):
    '''
    使用TF-IDF算法推荐图书
    :param book_comments:
    :return: 返回十本相似度最高的书籍
    '''
    if len(book_comments) > 0:
        # 保存字典
        print(len(book_comments))
        # 提取书名和评论文本
        book_names = []
        book_comms = []
        for book, comments in book_comments.items():
            book_names.append(book)
            book_comms.append(comments)
        # 构建TF-IDF特征矩阵
        vectorizer = TfidfVectorizer(stop_words=stop_words)
        tfidf_matrix = vectorizer.fit_transform([' '.join(comms) for comms in book_comms])
        # 计算图书之间的余弦相似度
        similarity_matrix = cosine_similarity(tfidf_matrix)
        book_idx = book_names.index(book_name)  # 获 取图书索引
        # 获 取与输入图书最相似的图书
        recommend_book_index = np.argsort(-similarity_matrix[book_idx])[1:11]
        # 输 出推荐的图书
        for idx in recommend_book_index:
            print(f"《{book_names[idx]}》 \t 相 似度： {similarity_matrix[book_idx][idx]:.4f}")
