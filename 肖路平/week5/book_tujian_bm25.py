import csv
import jieba

import numpy as np
from rank_bm25 import BM25Okapi

from bm25_code import bm25
def load_data(file_name,stop_words):
    #图书评论信息集合
    book_comments={} #{书名："评论1+评论+评论3+ 。。。"}

    with open(file_name, 'r',encoding='utf-8') as f:
        reader = csv.DictReader(f,delimiter='\t')
        for item in reader:
            book=item['book']
            comment=item['body']
            comment_words=jieba.lcut(comment)

            if book =="" : continue #跳过空书名

            filtered_comment_words = [word for word in comment_words]

            # 图书评论集合收藏
            book_comments[book]=book_comments.get(book, []) #[] 是当为空的时候，返回一个空列表（默认值是空列表）
            book_comments[book].extend(filtered_comment_words )
    return book_comments


if __name__ == '__main__':

    stop_words_file = r'e:\workspacepython\AiPremiumClass\肖路平\week5\stopwords.txt'
    stop_words=[line.strip() for line in open(stop_words_file,encoding='utf-8').readlines()]

    file_name = r'e:\workspacepython\AiPremiumClass\肖路平\week5\douban_comments_fixed.txt'
    book_comments=load_data(file_name,stop_words)
   
    
    book_names=[]
    book_comms=[]
    for book,comments in book_comments.items():
        book_names.append(book)
        book_comms.append(comments)

    bm25 = BM25Okapi(book_comms)   

    book_list = list(book_comments.keys())
    #print(book_list)
    book_name = input("请输入图书名：")

 
    # 获取输入书名的评论
    query_comments = book_comments[book_name]

    # 对查询评论进行分词并过滤停止词
    query_words = []
    for comment in query_comments:
        query_words.extend(jieba.lcut(comment))
    filtered_query_words = [word for word in query_words if word not in stop_words]

    # 获取所有图书的相似度分数
    scores = bm25.get_scores(filtered_query_words)

    # 创建一个包含书名和相似度分数的列表
    book_score_pairs = [(book_names[i], scores[i]) for i in range(len(book_names)) if book_names[i] != book_name]

    # 按相似度分数从高到低排序
    book_score_pairs.sort(key=lambda x: x[1], reverse=True)

  
    for book, score in book_score_pairs[:10]:  
        print(f"{book}: {score:.4f}")