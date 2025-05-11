import csv
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============ BM25 自定义实现 ============
def bm25(comments, k=1.5, b=0.75):
    N = len(comments)
    doc_lengths = []
    word_doc_freq = {}
    doc_term_dict = [{} for _ in range(N)]

    for i, comment in enumerate(comments):
        doc_lengths.append(len(comment))
        unique_words = set()
        for word in comment:
            doc_term_dict[i][word] = doc_term_dict[i].get(word, 0) + 1
            unique_words.add(word)
        for word in unique_words:
            word_doc_freq[word] = word_doc_freq.get(word, 0) + 1

    avg_doc_len = sum(doc_lengths) / N
    vocabulary = list(word_doc_freq.keys())
    word_index = {word: idx for idx, word in enumerate(vocabulary)}
    doc_term_matrix = np.zeros((N, len(vocabulary)))
    for i in range(N):
        for word, freq in doc_term_dict[i].items():
            idx = word_index.get(word)
            if idx is not None:
                doc_term_matrix[i, idx] = freq

    idf_numerator = N - np.array([word_doc_freq[word] for word in vocabulary]) + 0.5
    idf_denominator = np.array([word_doc_freq[word] for word in vocabulary]) + 0.5
    idf = np.log(idf_numerator / idf_denominator)
    idf[idf_numerator <= 0] = 0

    doc_lengths = np.array(doc_lengths)
    bm25_matrix = np.zeros((N, len(vocabulary)))
    for i in range(N):
        tf = doc_term_matrix[i]
        bm25_score = idf * (tf * (k + 1)) / (tf + k * (1 - b + b * doc_lengths[i] / avg_doc_len))
        bm25_matrix[i] = bm25_score

    # 构建每个评论的 BM25 向量
    final_bm25_matrix = []
    for i, comment in enumerate(comments):
        bm25_comment = []
        for word in comment:
            idx = word_index.get(word)
            if idx is not None:
                bm25_comment.append(bm25_matrix[i, idx])
        final_bm25_matrix.append(bm25_comment)

    max_len = max(len(row) for row in final_bm25_matrix)
    padded_matrix = [row + [0] * (max_len - len(row)) for row in final_bm25_matrix]
    return np.array(padded_matrix)

# 数据加载函数 
def load_data(filename):
    book_comments = {}
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for item in reader:
            book = item['book']
            comment = item['body']
            if book == '': continue
            words = [w for w in jieba.lcut(comment) if w.strip()]
            book_comments.setdefault(book, []).extend(words)
    return book_comments

#  主程序入口 
if __name__ == "__main__":
    # 1. 加载停用词
    stop_words = set(line.strip() for line in open("C:/Users/stephanie.chen/miniconda3/envs/py312/stopwords.txt", encoding='utf-8'))

    # 2. 加载评论数据
    book_comments = load_data("C:/Users/stephanie.chen/miniconda3/envs/py312/doubanbook_comments_fixed.txt")
    book_names = list(book_comments.keys())
    book_comms = list(book_comments.values())

    print(f"共加载 {len(book_names)} 本书")

    # 3. 构建 TF-IDF 模型
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(comments) for comments in book_comms])
    tfidf_sim_matrix = cosine_similarity(tfidf_matrix)

    # 4. 构建 BM25 相似度
    bm25_matrix = bm25(book_comms)
    bm25_sim_matrix = bm25_matrix @ bm25_matrix.T  # 点积作为相似度

    # 5. 用户输入推荐
    print("\n可选图书：")
    for i, name in enumerate(book_names):
        print(f"{i}. {name}")
    book_input = input("\n请输入你感兴趣的图书名称: ")
    if book_input not in book_names:
        print("书名不存在，请检查输入")
        exit()

    idx = book_names.index(book_input)

    print(f"\n基于 TF-IDF 推荐的图书（与《{book_input}》相似）：")
    tfidf_indices = np.argsort(-tfidf_sim_matrix[idx])[1:11]
    for i in tfidf_indices:
        print(f"{book_names[i]} \t 相似度: {tfidf_sim_matrix[idx][i]:.4f}")

    print(f"\n基于 BM25 推荐的图书（与《{book_input}》相似）：")
    bm25_indices = np.argsort(-bm25_sim_matrix[idx])[1:11]
    for i in bm25_indices:
        print(f"{book_names[i]} \t 相似度: {bm25_sim_matrix[idx][i]:.4f}")

    