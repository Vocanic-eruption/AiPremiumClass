import csv
import jieba
import numpy as np
from rank_bm25 import BM25Okapi

def load_data(douban_comments_fixed):
    # 图书评论信息集合
    book_comments = {}  
    with open(douban_comments_fixed, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')  
        for item in reader:
            book = item['book']
            comment = item['body']
            comment_words = jieba.lcut(comment)
            if book == '':
                continue
            if book not in book_comments:
                book_comments[book] = []
            book_comments[book].extend(comment_words)
    return book_comments

if __name__ == '__main__':
    # 加载停用词列表
    stop_words = set([line.strip() for line in open("stopwords.txt", "r", encoding="utf-8")])
    
    # 加载图书评论信息
    book_comments = load_data("douban_comments_fixed.txt")
    print("图书数量:", len(book_comments))
    
    # 预处理数据
    book_names = []
    tokenized_corpus = []
    for book, words in book_comments.items():
        # 过滤停用词
        filtered_words = [w for w in words if w not in stop_words and len(w) > 1]
        book_names.append(book)
        tokenized_corpus.append(filtered_words)
    
    # 构建BM25模型
    bm25 = BM25Okapi(tokenized_corpus)
    
    # 输入处理
    book_list = list(book_comments.keys())
    print("可用图书列表:", book_list)
    book_name = input("请输入图书名称：")
    query_idx = book_names.index(book_name)
    
    # 使用当前图书的所有词作为查询
    query = tokenized_corpus[query_idx]
    
    # 计算相似度得分
    doc_scores = bm25.get_scores(query)
    
    # 获取推荐索引（排除自身）
    recommend_indexes = np.argsort(doc_scores)[::-1][1:11]
    
    # 输出推荐结果
    print(f"\n基于BM25的推荐结果(最高显示10本):")
    for idx in recommend_indexes:
        score = doc_scores[idx]
        print(f"《{book_names[idx]}》\t相似度:{score:.4f}")