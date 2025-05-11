import numpy as np

def bm25(comments, k=1.5, b=0.75):
    """
    计算BM25分数矩阵。

    参数:
    comments (list of list of str): 文档列表，每个文档是一个词列表。
    k (float): BM25参数,通常设置为1.5。
    b (float): BM25参数,通常设置为0.75。

    返回:
    np.ndarray: 包含每个文档中每个词的BM25分数的矩阵。
    """
    # 计算文档总数
    N = len(comments)

    # 初始化文档长度列表和词频字典
    doc_lengths = [len(comment) for comment in comments]
    avg_doc_len = np.mean(doc_lengths)
    word_doc_freq = {}
    doc_term_dict = [{} for _ in range(N)]

    # 统计词频和包含该词的文档数量
    for i, comment in enumerate(comments):
        unique_words = set(comment)
        for word in comment:
            doc_term_dict[i][word] = doc_term_dict[i].get(word, 0) + 1
        for word in unique_words:
            word_doc_freq[word] = word_doc_freq.get(word, 0) + 1

    # 构建词汇表及其索引
    vocabulary = list(word_doc_freq.keys())
    word_index = {word: idx for idx, word in enumerate(vocabulary)}

    # 构建文档-词频矩阵
    doc_term_matrix = np.zeros((N, len(vocabulary)))
    for i in range(N):
        for word, freq in doc_term_dict[i].items():
            idx = word_index.get(word)
            if idx is not None:
                doc_term_matrix[i, idx] = freq

    # 计算idf值
    idf_numerator = N - np.array([word_doc_freq[word] for word in vocabulary]) + 0.5
    idf_denominator = np.array([word_doc_freq[word] for word in vocabulary]) + 0.5
    idf = np.log(idf_numerator / idf_denominator)
    idf[idf_numerator <= 0] = 0  # 避免出现 nan 值

    # 计算bm25值
    bm25_matrix = np.zeros((N, len(vocabulary)))
    for i in range(N):
        tf = doc_term_matrix[i]
        bm25_scores = idf * (tf * (k + 1)) / (tf + k * (1 - b + b * doc_lengths[i] / avg_doc_len))
        bm25_matrix[i] = bm25_scores

    # 重新排列bm25值以匹配原始评论的顺序，并填充到相同的长度
    final_bm25_matrix = []
    for i, comment in enumerate(comments):
        bm25_comment = [bm25_matrix[i, word_index[word]] for word in comment if word in word_index]
        final_bm25_matrix.append(bm25_comment)

    # 确保所有行具有相同的长度，不足的部分填充0
    max_length = max(len(row) for row in final_bm25_matrix)
    final_bm25_matrix = [row + [0] * (max_length - len(row)) for row in final_bm25_matrix]

    return np.array(final_bm25_matrix)

# 测试
if __name__ == '__main__':
    comments = [['a', 'b', 'c'], ['a', 'b', 'd'], ['a', 'b', 'e']]
    bm25_matrix = bm25(comments)
    print("BM25矩阵的形状:", bm25_matrix.shape)
    print("BM25矩阵:\n", bm25_matrix)
