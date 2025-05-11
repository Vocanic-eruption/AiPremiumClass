import jieba
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



def load_data(filename):

    book_comments = {} #书名 + 评论1 + 评论2 ......

    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for item in reader:
            book = item['book']
            comment = item['body']
            comment_words = jieba.lcut(comment)

            if book == '': continue
            book_comments[book] = book_comments.get(book, [])
            book_comments[book].extend(comment_words)
    return book_comments

if __name__ == '__main__':

    stop_words = [line.strip() for line in open("stopwords.txt", "r", encoding="utf-8")]

    book_comments = load_data("douban_comments_fixed.txt")

    book_names = []
    comments = []
    for key, val in book_comments.items():
        book_names.append(key)
        comments.append(val)

    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform([' '.join(comments) for comments in comments])

    similarity_matrix = cosine_similarity(tfidf_matrix)

    book_list = list(book_comments.keys())
    print(book_list)
    book_name = input("请输入图书名称：")
    book_idx = book_names.index(book_name)
    
    recommend_book_idx = np.argsort(-similarity_matrix[book_idx])[1:11]

    for i in recommend_book_idx:
        print(f"《{book_names[i]}》 \t 相似度：{similarity_matrix[book_idx][i]:.4f}")


