# 基于豆瓣top250图书评论的简单推荐系统
# 使用TF-IDF及BM25两种算法实现

import pandas as pd
import numpy as np
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import os

# 导入BM25算法
from bm25_code import bm25

# 设置文件路径
base_path = os.path.dirname(os.path.abspath(__file__))
douban_path = os.path.join(base_path, 'Douban_comments')
comments_file = os.path.join(douban_path, 'doubanbook_top250_comments.txt')
intro_file = os.path.join(douban_path, 'doubanbook_top250_introduction.txt')

# 读取数据
def load_data():
    # 读取评论数据，添加on_bad_lines='skip'参数跳过格式不正确的行
    comments_df = pd.read_csv(comments_file, sep='\t', on_bad_lines='skip')
    # 读取图书介绍数据
    intro_df = pd.read_csv(intro_file, sep='\t')
    return comments_df, intro_df

# 数据预处理
def preprocess_data(comments_df):
    # 去除评分为空的评论
    comments_df = comments_df[comments_df['star'] != 'none']
    
    # 将评分转换为数值
    star_map = {
        'allstar10': 1,
        'allstar20': 2,
        'allstar30': 3,
        'allstar40': 4,
        'allstar50': 5
    }
    comments_df['star_value'] = comments_df['star'].map(star_map)
    
    # 去除评论内容为空的行
    comments_df = comments_df[comments_df['body'].notna()]
    
    return comments_df

# 文本清洗
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 去除URL
    text = re.sub(r'http\S+', '', text)
    # 去除特殊字符和数字
    text = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 分词
def tokenize(text):
    text = clean_text(text)
    words = jieba.lcut(text)
    # 去除停用词（简单实现，实际应用中可以使用停用词表）
    words = [word for word in words if len(word) > 1]
    return words

# 基于TF-IDF的推荐系统
class TFIDFRecommender:
    def __init__(self, comments_df):
        self.comments_df = comments_df
        self.book_comments = {}
        self.tfidf_matrix = None
        self.vectorizer = None
        self.books = None
        
    def fit(self):
        # 按书名分组评论
        for book, group in self.comments_df.groupby('book'):
            self.book_comments[book] = ' '.join(group['body'].apply(clean_text))
        
        # 转换为列表
        self.books = list(self.book_comments.keys())
        corpus = [self.book_comments[book] for book in self.books]
        
        # 计算TF-IDF
        self.vectorizer = TfidfVectorizer(tokenizer=jieba.lcut, max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        
    def recommend(self, book_name, top_n=5):
        if book_name not in self.books:
            print(f"未找到书籍：{book_name}")
            return []
        
        # 获取书籍索引
        book_idx = self.books.index(book_name)
        
        # 计算相似度
        book_vector = self.tfidf_matrix[book_idx:book_idx+1]
        sim_scores = cosine_similarity(book_vector, self.tfidf_matrix).flatten()
        
        # 获取相似度最高的书籍（排除自身）
        sim_indices = sim_scores.argsort()[::-1]
        sim_indices = sim_indices[sim_indices != book_idx]
        top_indices = sim_indices[:top_n]
        
        # 返回推荐结果
        recommendations = [(self.books[i], sim_scores[i]) for i in top_indices]
        return recommendations

# 基于BM25的推荐系统
class BM25Recommender:
    def __init__(self, comments_df):
        self.comments_df = comments_df
        self.book_comments = {}
        self.book_tokens = {}
        self.similarity_matrix = None
        self.books = None
        
    def fit(self):
        # 按书名分组评论并分词
        for book, group in self.comments_df.groupby('book'):
            combined_text = ' '.join(group['body'].apply(clean_text))
            self.book_comments[book] = combined_text
            self.book_tokens[book] = tokenize(combined_text)
        
        # 转换为列表
        self.books = list(self.book_tokens.keys())
        tokenized_docs = [self.book_tokens[book] for book in self.books]
        
        # 计算BM25值
        bm25_matrix = bm25(tokenized_docs)
        
        # 计算文档间相似度
        self.similarity_matrix = np.zeros((len(self.books), len(self.books)))
        for i in range(len(self.books)):
            for j in range(len(self.books)):
                if i != j:
                    # 计算两个文档的BM25值的余弦相似度
                    vec1 = bm25_matrix[i]
                    vec2 = bm25_matrix[j]
                    # 确保向量长度相同
                    min_len = min(len(vec1), len(vec2))
                    vec1 = vec1[:min_len]
                    vec2 = vec2[:min_len]
                    # 计算余弦相似度
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    if norm1 > 0 and norm2 > 0:
                        self.similarity_matrix[i, j] = np.dot(vec1, vec2) / (norm1 * norm2)
        
    def recommend(self, book_name, top_n=5):
        if book_name not in self.books:
            print(f"未找到书籍：{book_name}")
            return []
        
        # 获取书籍索引
        book_idx = self.books.index(book_name)
        
        # 获取相似度最高的书籍
        sim_scores = self.similarity_matrix[book_idx]
        sim_indices = sim_scores.argsort()[::-1]
        top_indices = sim_indices[:top_n]
        
        # 返回推荐结果
        recommendations = [(self.books[i], sim_scores[i]) for i in top_indices]
        return recommendations

# 主函数
def main():
    print("加载数据...")
    comments_df, intro_df = load_data()
    
    print("预处理数据...")
    comments_df = preprocess_data(comments_df)
    
    print("\n基于TF-IDF的推荐系统：")
    tfidf_recommender = TFIDFRecommender(comments_df)
    print("训练TF-IDF模型...")
    tfidf_recommender.fit()
    
    print("\n基于BM25的推荐系统：")
    bm25_recommender = BM25Recommender(comments_df)
    print("训练BM25模型...")
    bm25_recommender.fit()
    
    # 测试推荐系统
    test_books = ['追风筝的人', '小王子', '活着']
    
    for book in test_books:
        print(f"\n为《{book}》推荐相似书籍：")
        
        print("\nTF-IDF推荐结果：")
        tfidf_recommendations = tfidf_recommender.recommend(book)
        for rec_book, score in tfidf_recommendations:
            print(f"- {rec_book} (相似度: {score:.4f})")
        
        print("\nBM25推荐结果：")
        bm25_recommendations = bm25_recommender.recommend(book)
        for rec_book, score in bm25_recommendations:
            print(f"- {rec_book} (相似度: {score:.4f})")

if __name__ == "__main__":
    main()
