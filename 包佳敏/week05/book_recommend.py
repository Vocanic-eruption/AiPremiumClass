import csv
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from BM25_code import bm25

def get_book_comments():
    book_comments = {} #创建一个字典，用于存储书籍评论,{书名：评论1词+评论2词+...}
    with open('comments_fixed.txt', 'r') as file:
        reader = csv.DictReader(file, delimiter='\t') #识别格式文本中的标题列
        i = 0
        for row in reader:
            book = row['book']
            comment = row['body']
            comment_words = jieba.cut(comment)

            if book == '' or comment == None: continue

            #将评论中的词添加到字典中
            book_comments[book] = book_comments.get(book, [])
            book_comments[book].extend(comment_words) #引用类型不是值类型
    return book_comments    

if __name__ == '__main__':
    book_comments = get_book_comments()
    #print(book_comments) #打印书籍评论
    #print(len(book_comments)) #打印书籍评论数量
    #for book, comments in book_comments.items():
    #    print(book, len(comments)) #打印书籍名字和评论数量
    #    print(comments) #打印评论
    #    break

    stop_words = [line.strip() for line in open('stop_words.txt', 'r', encoding='utf-8')] #读取停用词
    

    #创建一个TfidfVectorizer对象
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    #将评论转换为tf-idf矩阵
    tfidf_matrix = vectorizer.fit_transform(' '.join(comments) for comments in book_comments.values())
    #计算bm25矩阵
    final_bm25_matrix = bm25(book_comments.values())
    #计算余弦相似度
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    cosine_similarities_bm25 = cosine_similarity(final_bm25_matrix, final_bm25_matrix)
    print(cosine_similarities) #打印余弦相似度
    print(cosine_similarities.shape) #打印余弦相似度的形状
    print(cosine_similarities_bm25) #打印余弦相似度
    print(cosine_similarities_bm25.shape) #打印余弦相似度的形状

    book_names_result = []
    book_comments_result = []
    for book, comments in book_comments.items():
        book_names_result.append(book)
        book_comments_result.append(comments)

    book_name = input('请输入书名：')
    book_index = book_names_result.index(book_name)

    print('TF-IDF')
    recommand_book_index = np.argsort(-cosine_similarities[book_index])[1:11]
    for index in recommand_book_index:  
        print(book_names_result[index], cosine_similarities[book_index][index])

    print('BM25')
    recommand_book_index = np.argsort(-cosine_similarities_bm25[book_index])[1:11]  
    for index in recommand_book_index:  
        print(book_names_result[index], cosine_similarities_bm25[book_index][index])

