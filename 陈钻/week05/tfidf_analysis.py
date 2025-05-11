import csv 
import jieba 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
from tqdm import tqdm 

def load_tsv_data (filename): 
    book_comments = {}
    with open(filename, 'r') as f: 
        reader = csv.DictReader(f, delimiter='\t') 
        for item in reader: 
            book_name = item['book'] 
            comments = item['body']
            comment_words = jieba.lcut(comments)
            
            if book_name == "": continue
            
            book_comments[book_name] = book_comments.get(book_name, list()) 
            book_comments[book_name].extend(comment_words) 
    return book_comments


if __name__ == '__main__':
    file_name = 'doubanbook_top250_comments_fixed.txt'
    stopwords = [word.strip() for word in open('stopwords.txt' , 'r')]
    print('加载图书评论数据....')
    book_comments = load_tsv_data(file_name) 
    print('图书评论数据加载完毕！') 

    
    books_list = list(book_comments.keys()) 
    print('图书数量：' , len(books_list))
    print('构建TF-IDF矩阵...')
    vectorizer = TfidfVectorizer(stop_words=stopwords) 
    tfidf_matrix = vectorizer.fit_transform([' '.join(book_comments[book_name]) for book_name in books_list]) 
    print('TF-IDF矩阵构建完毕!')
    
    # 计算tfidf值的余弦相似度 
    similarities = cosine_similarity(tfidf_matrix) 
    
    # 找到与目标图书最相似的图书 
    print(books_list) 
    book_name = input("请输入图书名称：")
    book_idx = books_list.index(book_name) 
    
    # 取相似的排名前10的图书 
    recommend_book_idies = np.argsort(-similarities[book_idx][:11])[1:] 
    print('为您推荐的图书有：\n') 
    for idx in recommend_book_idies: 
        print(f"《{books_list[idx]}》 \t 相似度: {similarities[book_idx][idx]}")