import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

def bm25(comments, k=1.5, b=0.75):
    #计算文档总数
    N = len(comments)
    #初始化文档长度列表和字频字典
    doc_lengths = []
    word_doc_freq = {}
    doc_term_dict = [{} for _ in range(N)]

    for i, comment in enumerate(comments):
        #记录文档长度
        doc_lengths.append(len(comment))
        unique_words = set()
        for word in comment:
            #统计词频
            doc_term_dict[i][word] = doc_term_dict[i].get(word, 0) + 1
            unique_words.add(word)
        for word in unique_words:
            word_doc_freq[word] = word_doc_freq.get(word, 0) + 1
    #计算每个单词的平均文档长度
    avg_doc_len = sum(doc_lengths) / N
    #构建词汇表
    vocabulary = list(word_doc_freq.keys())
    word_index = {word:idx for indx, word in enumerate(vocabulary)}
    #构建文档  词频矩阵
    doc_term_matrix = np.zeros(N, len(vocabulary))
    for i in range(N):
        for word, freq in doc_term_dict[i].items():
            idx = word_index.get(word)
            if idx is not None:
                doc_term_matrix[i, idx] = freq
    #计算idf值
    idf_numerator = N - np.array([word_doc_freq[word] for word in vocabulary]) + 0.5
    idf_denominator = np.array([word_doc_freq[word] for word in vocabulary]) + 0.5
    idf = np.log(idf_numerator / idf_denominator)
    idf[idf_numerator <= 0] = 0 

    #计算bm25值
    doc_lengths = np.array(doc_lengths)
    bm25_matrix = np.zeros((N, len(vocabulary)))
    for i in range(N):
        tf = doc_term_matrix[i]
        bm25 = idf * (tf *(k + 1)) / (tf + k * (1-b+b*doc_lengths[i] / avg_doc_len))
        bm25_matrix[i] = bm25
    
    #根据原始评论顺序重新排列bm25
    final_bm25_matrix = []
    for i, comment in enumerate(comments):
        bm25_comment = []
        for word in comment:
            idx =word_index.get(word)
            if idx is not None:
                bm25_comment.append(bm25_matrix[i])
        final_bm25_matrix.append(bm25_comment)
    
    #找到最长的子列表长度
    max_length = max(len(row) for row in final_bm25_matrix)
    #填充所有子列表相同的长度
    padded_matrix = [row +[0]*(max_length - len(row)) for row in final_bm25_matrix]
    #装换为numpy数组
    final_bm25_matrix = np.array(padded_matrix)

    return final_bm25_matrix

from tf_idf import load_tsv_data

if __name__ == '__main__':
    filename = 'doubanbook_fixed.txt'
    book_comments = load_tsv_data(filename)
    bm_matrix = bm25(book_comments)
    similar_matrix = cosine_similarity(bm_matrix)

    book_list = list(book_comments.keys()) 
    input_name = input('请输入书名:')
    book_index = book_list.index(input_name) 
    recommend_book_idies = np.argsort(-similar_matrix[book_index])[:11][1:]
    
    for index in recommend_book_idies:
        print(book_list[index])
        print(f'相似度:{similar_matrix[book_index][index]}')
