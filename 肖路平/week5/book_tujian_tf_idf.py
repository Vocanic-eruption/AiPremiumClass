import csv
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
 

def load_data(file_name):
    #图书评论信息集合
    book_comments={} #{书名："评论1+评论+评论3+ 。。。"}

    with open(file_name, 'r',encoding='utf-8') as f:
        reader = csv.DictReader(f,delimiter='\t')
        for item in reader:
            book=item['book']
            comment=item['body']
            comment_words=jieba.lcut(comment)

            if book =="" : continue #跳过空书名
            # 图书评论集合收藏
            book_comments[book]=book_comments.get(book, []) #[] 是当为空的时候，返回一个空列表（默认值是空列表）
            book_comments[book].extend(comment_words)
    return book_comments

if __name__ == '__main__':

    file_name1 = r'e:\workspacepython\AiPremiumClass\肖路平\week5\stopwords.txt'
    stop_words=[line.strip() for line in open(file_name1,encoding='utf-8').readlines()]

    file_name2 = r'e:\workspacepython\AiPremiumClass\肖路平\week5\douban_comments_fixed.txt'
    book_comments=load_data(file_name2)
    print(len(book_comments))
    
    book_names=[]
    book_comms=[]
    for book,comments in book_comments.items():
        book_names.append(book)
        book_comms.append(comments)


    #构建TF-IDF 特征矩阵
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    #tfidf_matrix = vectorizer.fit_transform([' '.join(comments) for comments in book_comments.values()])
    #词表的填充，返回自身
    vectorizer.fit([' '.join(comments) for comments in book_comms])
    #评论词的得分矩阵
    tfidf_matrix=vectorizer.transform([' '.join(comments) for comments in book_comms])
    #fit 和 transform 是一个操作，fit_transform 是两个操作的组合
    print(tfidf_matrix.shape) #(218, 42702) 218个图书，42702个词



    print(vectorizer.get_feature_names_out())
    #余弦相似度计算
    #similarity_matrix 是通过余弦相似度计算得到的矩阵
    #矩阵中的每个元素 similarity_matrix[i][j] 表示第 i 本书和第 j 本书之间的余弦相似度。
    similarity_matrix = cosine_similarity(tfidf_matrix)#余弦角度，角度越小，相似越大

    print(similarity_matrix.shape)

    #输入推荐的图书名
    book_list = list(book_comments.keys())
    print(book_list)
    book_name = input("请输入图书名：")
    book_idx = book_list.index(book_name)#图书的索引
    
    recommend_book_index=np.argsort(-similarity_matrix[book_idx])[1:11]#只去前十本的相似度
    for idx in recommend_book_index:
         print(f"《{book_names[idx]}》 \t 相似度：{similarity_matrix[book_idx][idx]:.4f}")
 
    

 
