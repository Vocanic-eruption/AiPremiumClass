import csv
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import bm25_code
def load_data(fileName):
    book_comments={}

    with open(fileName,'r',encoding='utf-8') as f:
        reader=csv.DictReader(f,delimiter='\t')
        for item in reader:
            book=item['book']
            comment=item['body']
            comment_words=jieba.lcut(comment)
            if book=='':
                continue
            book_comments[book]=book_comments.get(book,[])
            book_comments[book].extend(comment_words)
    return book_comments

if __name__ == '__main__':
    
    stop_words=[line.strip() for line in open('week5/stopwords.txt','r',encoding='utf-8')]
    comments=load_data('doubanbook_comments.txt');
    book_names=[]
    book_comments=[]
    for book,comment in comments.items():
        book_names.append(book)
        book_comments.append(comment)
    book_list=list(comments.keys())
    print(book_list)
    book_name=input("请输入图书名称:")
    book_idx=book_names.index(book_name)
    
    
    vectorizer=TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix=vectorizer.fit_transform([' '.join(comments) for comments in book_comments])
    similarity_matrix=cosine_similarity(tfidf_matrix)
    recommand_book_index=np.argsort(-similarity_matrix[book_idx][1:11])
    for idx in recommand_book_index:
        print(f"TF-IDF：《{book_names[idx]}》\t相似度：{similarity_matrix[book_idx][idx]:.4f}")
    print()
    
    bm25_matrix = bm25_code.bm25(book_comments);
    similarity_matrix=cosine_similarity(bm25_matrix)
    recommand_book_index=np.argsort(-similarity_matrix[book_idx][1:11])
    for idx in recommand_book_index:
        print(f"BM25：《{book_names[idx]}》\t相似度：{similarity_matrix[book_idx][idx]:.4f}")
    print()
    
    