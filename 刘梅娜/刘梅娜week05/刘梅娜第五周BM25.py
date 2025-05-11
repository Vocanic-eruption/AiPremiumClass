from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd

# 示例数据
data = {
    'book_title': ['Book1', 'Book2', 'Book3'],
    'comments': [
        'This is a great book about science.',
        'A wonderful novel with a deep plot.',
        'Science fiction at its best.'
    ]
}
df = pd.DataFrame(data)

# 分词处理
corpus = [comment.split() for comment in df['comments']]

# 初始化BM25模型
bm25 = BM25Okapi(corpus)

def bm25_recommendations(query, top_n=5):
    tokenized_query = query.split()
    doc_scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(doc_scores)[::-1][:top_n]
    return df['book_title'].iloc[top_indices]

# 示例推荐
print(bm25_recommendations('science'))
