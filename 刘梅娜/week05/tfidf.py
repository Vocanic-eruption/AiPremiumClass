import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 假设我们有一个包含豆瓣top250图书评论的DataFrame
# df = pd.read_csv('douban_top250_comments.csv')

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

# 使用TF-IDF向量化评论
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['comments'])

# 计算余弦相似度
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# 构建索引映射
indices = pd.Series(df.index, index=df['book_title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    book_indices = [i[0] for i in sim_scores]
    return df['book_title'].iloc[book_indices]

# 示例推荐
print(get_recommendations('Book1'))
