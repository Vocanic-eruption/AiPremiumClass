import numpy as np
import jieba
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(file_name):
    book_comments = {}
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            book = row['book']
            comments = row['body']
            if book == '' or comments is None:
                continue
            comment_words = jieba.lcut(comments)
            # print(comment_words)
            book_comments[book] = book_comments.get(book, [])
            book_comments[book].extend(comment_words)
    return book_comments


if __name__ == '__main__':
    book_comments = load_data('./comments_fix.txt')
    # print(len(book_comments))
    # print(book_comments.values())
    stop_words = [word.strip() for word in open("./stopwords.txt", "r", encoding="utf-8")]

    vectorizer = TfidfVectorizer(stop_words=stop_words)
    # aa = [' '.join(comments) for comments in book_comments.values()]
    # tfidf_matrix = vectorizer.fit_transform(aa)
    books_name = []
    books_comment = []
    for book, comments in book_comments.items():
        books_name.append(book)
        books_comment.append(comments)
    tfidf_matrix = vectorizer.fit_transform(''.join(bc) for bc in books_comment)
    print(tfidf_matrix.shape)
    similarities_matrix = cosine_similarity(tfidf_matrix)
    # print(similarities_matrix.shape)
    # print(type(book_comments.keys()))
    book_list = list(book_comments.keys())
    print(book_list)
    book_name = input("请输入图书名称：")
    book_idx = books_name.index(book_name)
    recommend_book_index = np.argsort(-similarities_matrix[book_idx])[1:11]
    for idx in recommend_book_index:
        print(f"《{books_name[idx]}》 \t 相似度：{similarities_matrix[book_idx][idx]:.4f}")

