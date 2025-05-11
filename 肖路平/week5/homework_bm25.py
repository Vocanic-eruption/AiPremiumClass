import csv
import jieba

import numpy as np
from bm25_code import bm25
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
def comment_vectors_similarity(book_comms,method='bm25'):
    if method=='tfidf':
        vectorizer=TfidfVectorizer()
        matrix=vectorizer.fit_transform([' '.join(comments) for comments in book_comms])
    if method=='bm25':
        matrix=bm25(book_comms)
    
    similarty_matrix=cosine_similarity(matrix)
    return similarty_matrix

if __name__=='__main__':
    stop_words=[line.strip() for line in open('stop_words.txt','r',encoding='utf-8')]

    book_comments=load_data('book_comments.txt',stop_words)

    book_names=[]
    book_comms=[]

    for book,comments in book_comments.items():
        book_names.append(book)
        book_comms.append(comments)
    
    tfidf_matrix=comment_vectors_similarity(book_comms,method='tfidf')

    bm25_martix = comment_vectors_similarity(book_comms,method='bm25')
 