import csv
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import BM25

def load_data(filename):
    #{书名:"评论1词+评论2词+...}
    book_comments = {}

    with open('D://ai//badou//codes//第五周//fixed_comments.txt', 'r',encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for item in reader:
            book = item['book']
            comment = item["body"]
            comment_words = jieba.lcut(comment)
            
            if book == '': continue

            #t图书评论集合收集
            book_comments[book] = book_comments.get(book,[])
            book_comments[book].extend(comment_words) 

    return book_comments

if __name__ == '__main__':
    #加载停用词
    stop_words = [line.strip() for line in open('D://ai//badou//codes//第五周//stopwords.txt', 'r', encoding='utf-8').readlines()]
    
    #加载图书评论
    book_comments = load_data('D://ai//badou//codes//第五周//fixed_comments.txt')
    
    book_names = []
    book_comments_list = []
    for book,comments in book_comments.items():
        book_names.append(book)
        book_comments_list.append(comments)

    #构建TF-IDF特征矩阵
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform([' '.join(comments) for comments in book_comments.values()])

    #BM25
    bm25_matrix = BM25.bm25(book_comments_list)

    #计算图书之间的相似度
    tfidf_similarity_matrix = cosine_similarity(tfidf_matrix)

    # 计算图书之间的相似度
    bm25_similarity_matrix = cosine_similarity(bm25_matrix)

    book_list = list(book_comments.keys())
    print(book_list)
    book_name = input("请输入图书名称：")
    book_index = book_names.index(book_name)

    recommend_book_index =  np.argsort(-tfidf_similarity_matrix[book_index])[1:11]

    for i in recommend_book_index:
        print(f" 《{book_names[i]}》 \t 相似度：{tfidf_similarity_matrix[book_index][i]}")

    print("-------------以下是BM25-----------------")

    recommend_book_index =  np.argsort(-bm25_similarity_matrix[book_index])[1:11]

    for i in recommend_book_index :
        print(f" 《{book_names[i]}》 \t 相似度：{bm25_similarity_matrix[book_index][i]}")


