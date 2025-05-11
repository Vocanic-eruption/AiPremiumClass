from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rank_bm25 import BM25Okapi
import jieba

with open(r"D:\work\code\practice\home_work\杨文博\week05\doubanbook_top250_comments.txt", 'r',encoding="utf-8") as file:
    lines = file.readlines()

books_comments = defaultdict(list)
last_book_name = lines[1].split("\t")[0]
for line in lines[1:]:
    line = line.strip("\n")
    x = line.split("\t")
    if len(x) != 6:
        books_comments[last_book_name][-1] += "".join(x)
    else:
        if "allstar" in x[2]:
            books_comments[x[0]].append(x[-1])
            last_book_name = x[0]
        else:
            books_comments[last_book_name][-1] += "".join(x)

books_comments_str = {key: "".join(value) for key, value in books_comments.items()}
stop_words = [line.strip() for line in open(r'D:\work\code\practice\home_work\杨文博\week05\stopwords.txt', 'r', encoding="utf-8")]

book_names = []
book_comms = []
for k,v in books_comments_str.items():
    book_names.append(k)
    book_comms.append(v)

# # 构建tfidf矩阵
# vectorizer = TfidfVectorizer(stop_words=stop_words)
# tfidf_matrix = vectorizer.fit_transform(book_comms)
# cs_similarity = cosine_similarity(tfidf_matrix)


#构建bm25
def tokenize_and_filter(text):
    words = list(jieba.cut(text))
    return [word for word in words if word not in stop_words and len(word) > 1]  # 去除停用词和单字

tokenized_corpus = [tokenize_and_filter(doc) for doc in book_comms]

bm25 = BM25Okapi(tokenized_corpus)
bm25_matrix = np.array([bm25.get_scores(doc) for doc in tokenized_corpus])
cs_similarity = cosine_similarity(bm25_matrix)

print(book_names)
while True:
    book_like = input("输入书名")
    if book_like in book_names:
        break
book_like_index = book_names.index(book_like)

recommend_book_index = np.argsort(-cs_similarity[book_like_index])[1:11]
for idx in recommend_book_index:
    print(book_names[idx],"\t",cs_similarity[book_like_index][idx])

#构建bm25矩阵
tokenized_corpus = [list(jieba.cut(doc)) for doc in book_comms]
bm25 = BM25Okapi(tokenized_corpus)
bm25_matrix = np.array([bm25.get_scores(doc) for doc in tokenized_corpus])
cs_similarity = cosine_similarity(bm25_matrix)
