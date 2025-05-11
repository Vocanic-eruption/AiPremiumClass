import csv 
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
import io

def load_tsv_data(filename):
    #用来将所有书的评论合并到一块
    book_comments={}
    with open(filename,'r',encoding='utf-8') as f:
        #按标题进行读取，第一行会被读成标题
        reader=csv.DictReader(f,delimiter='\t')
        #此处每行相当于被读成item中一行
        for item in reader:
            book_name=item['book']
            comments=item['body']
            if book_name=="":
                continue
            comments_words=jieba.lcut(comments)
           #此处使用get方法将book_comment原本的元素全拿出来,get失败会返回空列表也对
            book_comments[book_name]=book_comments.get(book_name,list())
            book_comments[book_name].extend(comments_words)
        return book_comments
#除了创建评估矩阵部分不一致，其他部分是一致的，引用bm25F算法

def bm25(comments, k=1.5, b=0.75):
    # 计算文档总数
    N = len(comments)
    # 初始化文档长度列表和词频字典
    doc_lengths = []
    word_doc_freq = {}
    doc_term_dict = [{} for _ in range(N)]

    for i, comment in enumerate(comments):
        # 记录文档长度
        doc_lengths.append(len(comment))
        unique_words = set()
        for word in comment:
            # 统计第i个文档之中，某个word的词频
            doc_term_dict[i][word] = doc_term_dict[i].get(word, 0) + 1
            unique_words.add(word)
        # 统计包含该词的文档数量
        for word in unique_words:
            word_doc_freq[word] = word_doc_freq.get(word, 0) + 1

    # 计算每个单词的平均文档长度（总文档长度除以文档总数）
    avg_doc_len = sum(doc_lengths) / N

    # 构建词汇表
    vocabulary = list(word_doc_freq.keys())
    #构建字典，将词和对应的vocabulary下标放进来
    word_index = {word: idx for idx, word in enumerate(vocabulary)}
    # 构建文档 - 词频矩阵
    doc_term_matrix = np.zeros((N, len(vocabulary)))
    #在N个文档之中
    for i in range(N):
        for word, freq in doc_term_dict[i].items():
            idx = word_index.get(word)
            if idx is not None:
                doc_term_matrix[i, idx] = freq

    # 计算 idf 值
    idf_numerator = N - np.array([word_doc_freq[word] for word in vocabulary]) + 0.5
    idf_denominator = np.array([word_doc_freq[word] for word in vocabulary]) + 0.5
    idf = np.log(idf_numerator / idf_denominator)
    idf[idf_numerator <= 0] = 0  # 避免出现 nan 值

    # 计算 bm25 值
    doc_lengths = np.array(doc_lengths)
    bm25_matrix = np.zeros((N, len(vocabulary)))
    for i in range(N):
        tf = doc_term_matrix[i]
        bm25 = idf * (tf * (k + 1)) / (tf + k * (1 - b + b * doc_lengths[i] / avg_doc_len))
        bm25_matrix[i] = bm25

    # 根据原始评论顺序重新排列 bm25 值
    final_bm25_matrix = []
    for i, comment in enumerate(comments):
        bm25_comment = []
        for word in comment:
            idx = word_index.get(word)
            if idx is not None:
                bm25_comment.append(bm25_matrix[i, idx])
        final_bm25_matrix.append(bm25_comment)

    # 找到最长的子列表长度
    max_length = max(len(row) for row in final_bm25_matrix)
    # 填充所有子列表到相同的长度
    padded_matrix = [row + [0] * (max_length - len(row)) for row in final_bm25_matrix]
    # 转换为 numpy 数组
    final_bm25_matrix = np.array(padded_matrix)

    return final_bm25_matrix

if __name__=='__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') 
    file_name="doubanbook_top251_comments_fixed.txt"
    
    print("加载图书评论数据")
    book_comments=load_tsv_data(file_name)

    print("图书数据加载完毕")
    book_list=list(book_comments.keys())#拿出所有书名
    print('图书数量:',book_list)

    print('构建TF-IDF矩阵...')
   
    #此处comments参数是一个大列表，因此我们改变一下参数导入方法
    comments=list(book_comments.values())
    tfidf_matrix=bm25(comments)
    #计算tf余弦相似度
    simlarities=cosine_similarity(tfidf_matrix)
    print(simlarities)

    #找到与目标图书最相似的图书
    book_name=input("请输入图书名称: ")
    #将书名转化为对应索引进入矩阵取搜索
    book_idx=book_list.index(book_name)

    #取相似前10排名的图书(除去自己)
    recommend_book_idies=np.argsort(-simlarities[book_idx][:11][1:])
    print("为您推荐的图书有：")
    for idx in recommend_book_idies:
        print(f'{book_list[idx]}\t 相似度:{simlarities[book_idx][idx]}')
'''
字符串拼接
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
book_comments = {
'book1': ['这是第一本书的评论1', '这是第一本书的评论2'],
'book2': ['这是第二本书的评论1', '这是第二本书的评论2']
}
book_list = ['book1', 'book2']
corpus = [''.join(book_comments[book_name]) for book_name in book_list]
print(corpus)
'''
