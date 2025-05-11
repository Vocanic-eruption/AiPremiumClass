
import bm25_code
import my_book_recommend
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

file_path2 = r"./王健钢\week5_语言模型\douban_comments_fixed.txt"
# 加载图书评论信息
book_comments = my_book_recommend.load_data(file_path2)
print(len(book_comments))
# 提取书名和评论文本
book_names = []
book_comms = []
for book, comments in book_comments.items():
    book_names.append(book)
    book_comms.append(comments)
# 构建BM25特征矩阵
bm25_matrix = bm25_code.bm25(book_comms)
print(bm25_matrix)

# 计算图书之间的余弦相似度
similarity_matrix = cosine_similarity(bm25_matrix)

# 输入要推荐的图书名称
book_list = list(book_comments.keys())
print(book_list)
book_name = input("请输入图书名称：")
book_idx = book_names.index(book_name)  # 获取图书索引

# 获取与输入图书最相似的图书
recommend_book_index = np.argsort(-similarity_matrix[book_idx])[1:11]
# 输出推荐的图书
for idx in recommend_book_index:
    print(f"《{book_names[idx]}》 \t 相似度：{similarity_matrix[book_idx][idx]:.4f}")
print()