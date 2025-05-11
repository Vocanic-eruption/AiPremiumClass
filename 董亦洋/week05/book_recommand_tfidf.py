import csv
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#图书评论集合 douban_comments_fixed.txt
def get_comments(file_name):
    book_comments = {} #{书名：全部评论分词}
    with open(file_name ,"r",encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for item in reader:
            book = item['book']
            comment = item['body']
            if book==''or comment==None: continue

            comment_words = jieba.lcut(comment)

            if (book_comments.get(book)==None):
                book_comments[book] = comment_words
            else:
                book_comments[book].extend(comment_words)
    return book_comments
        
if __name__ =='__main__':
    #加载停用词列表
    stop_words = [line.strip() for line in open('stopwords.txt','r',encoding='utf-8')]
    #加载图书评论信息
    all_book_comments = get_comments('douban_comments_fixed.txt')
    #提取书名和评论
    book_names = []
    book_comments = [] 
    for k,v in all_book_comments.items():
        book_names.append(k)
        book_comments.append(v)
    #构建TF_IDF特征矩阵
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform([' '.join(comments) for comments in book_comments])
    #计算余弦相似度
    similarity = cosine_similarity(tfidf_matrix)

    #输入图书名称
    print(book_names)
    book_name = input("请输入图书名称")
    book_idx = book_names.index(book_name)

    recommand_list = np.argsort(-similarity[book_idx])[1:11]
    for i in recommand_list:
        print(f'书名：《{book_names[i]}》，相似度：{similarity[book_idx][i]}')

