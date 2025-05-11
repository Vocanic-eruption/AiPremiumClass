from load_comments import load_comments
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 加载停用词列表
stop_words = [lines.strip() for lines in open('stopwords.txt', encoding='utf-8')]
stop_words.extend(['ain', 'aren', 'couldn', 'daren', 'didn',
                   'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn',
                   'll', 'mayn', 'mightn', 'mon', 'mustn', 'needn',
                   'oughtn', 'shan', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn'])

fixed_file = 'doubanbook_top250_comments_fixed.txt'
book_comments = {}

# 加载评论数据
if not os.path.exists('book_comments.npy'):
    book_comments = load_comments(fixed_file)
    # print("book_names:", book_comments.keys())
    np.save('book_comments.npy', book_comments)
else:
    print("Loading book_comments.npy...")
    book_comments = np.load('book_comments.npy', allow_pickle=True).item()
    # print("book_names:", book_comments.keys())
    # print("len(book_comments):", len(book_comments)) # 237

# 提取书名和评论文本
book_names = []
comment_texts = []
for book, comments in book_comments.items():
    book_names.append(book)
    comment_texts.append(comments)

# print(f"{comment_texts[0][:10]}") # ['带', '着', '了解', '精神病', '群体', '的', '期待', '看', '这', '本书']

# 计算TF-IDF
vectorizer = TfidfVectorizer(stop_words=stop_words)
tfidf_matrix = vectorizer.fit_transform([' '.join(comment) for comment in comment_texts])

# print(tfidf_matrix.toarray().shape) # (237, 73727)
# print(tfidf_matrix.shape) # (237, 73727)

# 计算图书之间的余弦相似度
cosine_sim = cosine_similarity(tfidf_matrix)

# print(f"cosine_sim.shape: {cosine_sim.shape}") # (237, 237)

# sub = np.array([row[:5] for row in cosine_sim[:5]])
# sub = cosine_sim[:5, :5]
# print(f"{sub}")

# 输入要推荐的书名
book_list = list(book_comments.keys())
print(f"book_list: {book_list}")
book_name =input(f"请输入图书名称: ")
book_idx = book_list.index(book_name)

# 获取与输入书籍最相似的10个书籍
recommend_idx = np.argsort(cosine_sim[book_idx])[::-1][1:11]
recommend_books = [book_list[i] for i in recommend_idx]

# 输出推荐的图书
for i, book in enumerate(recommend_books):
    print(f"《{book}》", f"相似度: {cosine_sim[book_idx][recommend_idx[i]]:.4f}")

print()

