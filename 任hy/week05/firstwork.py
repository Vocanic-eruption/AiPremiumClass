import csv
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rank_bm25 import BM25Okapi

def load_data(filename):
    # 图书评论信息集合
    book_comments = {}  # {书名: ["评论1", "评论2", ...]}
    book_info = {}      # {书名: {"author": 作者, "rating": 评分}}

    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')  # 识别格式文本中标题列
        for item in reader:
            book = item['book'].strip()
            author = item['author'].strip()
            rating = item['rating'].strip()
            comment = item['body'].strip()
            
            if not book:  # 跳过空书名
                continue
                
            # 保存图书基本信息
            if book not in book_info:
                book_info[book] = {"author": author, "rating": rating}
            
            # 图书评论集合收集
            if book not in book_comments:
                book_comments[book] = []
            if comment:  # 只添加非空评论
                book_comments[book].append(comment)
    
    return book_comments, book_info

def preprocess_text(text, stop_words):
    # 分词并去除停用词
    words = jieba.lcut(text)
    return [word for word in words if word not in stop_words and word.strip()]

def tfidf_recommend(book_name, book_names, tfidf_matrix, top_n=10):
    """基于TF-IDF和余弦相似度的推荐"""
    if book_name not in book_names:
        print("未找到该图书，请检查输入是否正确。")
        return []
    
    book_idx = book_names.index(book_name)
    similarity_scores = cosine_similarity(tfidf_matrix[book_idx], tfidf_matrix).flatten()
    related_indices = np.argsort(-similarity_scores)[1:top_n+1]  # 排除自己
    
    recommendations = []
    for idx in related_indices:
        recommendations.append({
            'book': book_names[idx],
            'similarity': similarity_scores[idx]
        })
    
    return recommendations

def bm25_recommend(book_name, book_names, tokenized_corpus, bm25, top_n=10):
    """基于BM25算法的推荐"""
    if book_name not in book_names:
        print("未找到该图书，请检查输入是否正确。")
        return []
    
    book_idx = book_names.index(book_name)
    scores = bm25.get_scores(tokenized_corpus[book_idx])
    related_indices = np.argsort(-scores)[1:top_n+1]  # 排除自己
    
    recommendations = []
    for idx in related_indices:
        recommendations.append({
            'book': book_names[idx],
            'score': scores[idx]
        })
    
    return recommendations

def print_recommendations(recommendations, book_info, algorithm_name):
    """打印推荐结果"""
    print(f"\n基于{algorithm_name}算法的推荐结果：")
    print("-" * 60)
    print(f"{'序号':<5}{'图书名称':<30}{'作者':<15}{'评分':<8}{'相似度/分数':<10}")
    print("-" * 60)
    
    for i, rec in enumerate(recommendations, 1):
        book = rec['book']
        info = book_info.get(book, {"author": "未知", "rating": "无"})
        score = rec.get('similarity', rec.get('score', 0))
        
        # 格式化输出
        print(f"{i:<5}{book:<30}{info['author']:<15}{info['rating']:<8}{score:.4f}")

if __name__ == '__main__':
    # 加载停用词列表
    stop_words = set()
    with open("stopwords.txt", "r", encoding="utf-8") as f:
        for line in f:
            stop_words.add(line.strip())
    
    # 加载图书评论信息和基本信息
    book_comments, book_info = load_data("douban_comments_fixed.txt")
    print(f"加载完成，共 {len(book_comments)} 本图书的评论数据。")
    
    # 准备数据
    book_names = []
    processed_comments = []
    tokenized_corpus = []
    
    for book, comments in book_comments.items():
        book_names.append(book)
        # 合并所有评论为一个文档
        full_text = ' '.join(comments)
        # 预处理文本
        processed = preprocess_text(full_text, stop_words)
        processed_comments.append(' '.join(processed))
        tokenized_corpus.append(processed)
    
    # 1. TF-IDF推荐
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_comments)
    
    # 2. BM25推荐
    bm25 = BM25Okapi(tokenized_corpus)
    
    # 用户交互
    while True:
        print("\n当前可推荐的图书列表：")
        for i, book in enumerate(book_names, 1):
            print(f"{i}. {book}")
        
        print("\n请输入要查询的图书名称或编号（输入q退出）：")
        user_input = input().strip()
        
        if user_input.lower() == 'q':
            break
        
        # 处理数字输入
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(book_names):
                book_name = book_names[idx]
            else:
                print("编号超出范围，请重新输入。")
                continue
        else:
            book_name = user_input
        
        if book_name not in book_names:
            print("未找到该图书，请检查输入是否正确。")
            continue
        
        # 获取推荐
        tfidf_recs = tfidf_recommend(book_name, book_names, tfidf_matrix)
        bm25_recs = bm25_recommend(book_name, book_names, tokenized_corpus, bm25)
        
        # 打印推荐结果
        print(f"\n您选择的图书是：《{book_name}》")
        info = book_info.get(book_name, {"author": "未知", "rating": "无"})
        print(f"作者：{info['author']}，评分：{info['rating']}")
        
        print_recommendations(tfidf_recs, book_info, "TF-IDF")
        print_recommendations(bm25_recs, book_info, "BM25")
