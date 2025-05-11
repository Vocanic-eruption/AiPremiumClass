from rank_bm25 import BM25Okapi
import jieba
import csv
import numpy as np

def load_data(filename):
    """加载数据并返回图书评论字典（书名: 分词后的评论列表）"""
    book_comments = {}
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for item in reader:
            if len(item) < 6 or item[0] == 'book':  # 跳过标题行和无效数据
                continue
            book_name, comment = item[0], item[5]
            if book_name not in book_comments:
                book_comments[book_name] = []
            # 分词并过滤单字词
            words = [w for w in jieba.lcut(comment) if len(w) > 1]
            book_comments[book_name].extend(words)
    return book_comments

def build_bm25_model(book_comments, stopwords=None):
    """构建BM25模型"""
    book_names = list(book_comments.keys())
    # 过滤停用词
    tokenized_corpus = []
    for name in book_names:
        if stopwords:
            words = [w for w in book_comments[name] if w not in stopwords]
        else:
            words = book_comments[name]
        tokenized_corpus.append(words)
    return BM25Okapi(tokenized_corpus), book_names, tokenized_corpus  # 返回分词后的语料

def find_similar_books(bm25_model, book_names, tokenized_corpus, query_name, top_n=5):
    """查找相似书籍"""
    query_idx = book_names.index(query_name)
    # 使用书籍自身评论作为查询
    scores = bm25_model.get_scores(tokenized_corpus[query_idx])  # 直接使用分词后的查询
    top_indices = np.argsort(scores)[-top_n-1:-1][::-1]  # 排除自己
    results = []
    for idx in top_indices:
        results.append((book_names[idx], float(scores[idx])))
    return results

if __name__ == '__main__':
    # 1. 加载数据
    data_path = './党金虎/week05/douban_test_data/doubanbook_top250_comments_fixed.txt'
    book_comments = load_data(data_path)
    print(f"加载完成，共有 {len(book_comments)} 本图书")

    # 2. 加载停用词
    stopwords_path = './党金虎/week05/douban_test_data/stopwords.txt'
    stopwords = set(line.strip() for line in open(stopwords_path, encoding='utf-8'))
    print(f"加载 {len(stopwords)} 个停用词")

    # 3. 构建BM25模型（同时获取分词后的语料）
    bm25_model, book_names, tokenized_corpus = build_bm25_model(book_comments, stopwords)
    print("BM25模型构建完成")

    # 4. 交互式查询
    while True:
        print("\n当前图书列表前20本:", book_names[:20])
        query = input("请输入书名（输入q退出）:").strip()
        if query.lower() == 'q':
            break
        if query not in book_names:
            print("未找到该书，请检查书名！")
            continue
        
        # 查找相似书籍（传入分词后的语料）
        similar_books = find_similar_books(bm25_model, book_names, tokenized_corpus, query)
        print(f"\n与《{query}》最相似的 {len(similar_books)} 本书：")
        for i, (name, score) in enumerate(similar_books, 1):
            print(f"{i}. {name} (相似度: {score:.4f})")