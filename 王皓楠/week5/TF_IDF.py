import csv 
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
import io
def load_tsv_data(filename):
    #用来将所有书的评论合并到一块
    book_comments={}
    with open(filename,'r',encoding='utf-8') as f:
        #按标题进行读取，第一行会被读成标题
        reader=csv.DictReader(f,delimiter='\t')
        #此处每行相当于被读成item中一行
        for item in reader:
            book_name=item['book']
            comments=item['body']
            if book_name=="":
                continue
            comments_words=jieba.lcut(comments)
           #此处使用get方法将book_comment原本的元素全拿出来,get失败会返回空列表也对
            book_comments[book_name]=book_comments.get(book_name,list())
            book_comments[book_name].extend(comments_words)
        return book_comments

if __name__=='__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') 
    file_name="doubanbook_top251_comments_fixed.txt"
    stopwords=[word.strip() for word in open('stopwords.txt','r',encoding='utf-8')]
    print("加载图书评论数据")
    book_comments=load_tsv_data(file_name)

    print("图书数据加载完毕")
    book_list=list(book_comments.keys())#拿出所有书名
    print('图书数量:',book_list)

    print('构建TF-IDF矩阵...')
    vectorizer=TfidfVectorizer(stop_words=stopwords)
    tfidf_matrix=vectorizer.fit_transform(' '.join(book_comments[book_name]) for book_name in book_list)
    #计算tf余弦相似度
    simlarities=cosine_similarity(tfidf_matrix)
    print(simlarities)

    #找到与目标图书最相似的图书
    book_name=input("请输入图书名称: ")
    #将书名转化为对应索引进入矩阵取搜索
    book_idx=book_list.index(book_name)

    #取相似前10排名的图书(除去自己)
    recommend_book_idies=np.argsort(-simlarities[book_idx][:11][1:])
    print("为您推荐的图书有：")
    for idx in recommend_book_idies:
        print(f'{book_list[idx]}\t 相似度:{simlarities[book_idx][idx]}')
'''
字符串拼接
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
book_comments = {
'book1': ['这是第一本书的评论1', '这是第一本书的评论2'],
'book2': ['这是第二本书的评论1', '这是第二本书的评论2']
}
book_list = ['book1', 'book2']
corpus = [''.join(book_comments[book_name]) for book_name in book_list]
print(corpus)
'''
