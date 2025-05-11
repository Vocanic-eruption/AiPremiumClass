import csv # 用于读取csv文件，处理表格数据。
import jieba # 用于中文分词，将文本切分成词语。
from sklearn.feature_extraction.text import TfidfVectorizer # 用于构建TF-IDF特征矩阵，用于计算文本特征。
from sklearn.metrics.pairwise import cosine_similarity # 用于计算余弦相似度，用于计算文本之间的相似度。
from bm25_code import bm25 # 用于构建BM25模型，用于计算文本之间的相似度。
import numpy as np # 用于处理数组和矩阵数据。


def load_data(filename):
    # 图书评论合集
    book_commemts = {} # {书名：“评论1\n评论2\n评论3\n”}

    with open(filename,'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t') # 读取csv文件
        for item in reader: # 遍历每一行
            book = item['book'] # 提取书名
            comment = item['body'] # 提取评论
            comment_words = jieba.lcut(comment) # 分词


            if book == '': # 书名为空，跳过
                continue

            # 图书评论集合收集
            book_commemts[book] = book_commemts.get(book, [])
            book_commemts[book].extend(comment_words) # 收集评论

    return book_commemts

if __name__ == '__main__':

    # 加载停用词
    stop_words = [line.strip() for line in open('stopwords.txt', 'r', encoding='utf-8')]

    # 加载数据
    book_commemts = load_data('douban_comments_fixed.txt')
    print(book_commemts)

    # 提取书名和评论文本
    book_names = []
    book_comms = []
    for book, comments in book_commemts.items():
        book_names.append(book) # 提取书名
        book_comms.append(comments) # 提取评论

    # # 构建TF-IDF特征矩阵
    # vectorizer = TfidfVectorizer(stop_words=stop_words) # 初始化TF-IDF向量器
    # tfidf_matrix = vectorizer.fit_transform(' '.join(comms) for comms in book_comms) # 构建TF-IDF特征矩阵

    # # 计算图书之间的余弦相似度
    # similarity_matrix = cosine_similarity(tfidf_matrix) # 计算余弦相似度矩阵，用于计算图书之间的相似度。

    # 初始化BM25模型
    bm25_model  = bm25(book_comms) # 初始化BM25模型，用于计算图书之间的相似度。

    # 输入要推荐的图书名称
    book_list = list(book_commemts)
    print(book_list) # 打印所有的图书名称，用于选择要推荐的图书。可以根据需要选择要推荐的图书。
    book_name = input("请输入要推荐的图书：")
    book_idx = book_names.index(book_name) # 获取图书索引

    # 获取与输入图书最相似的图书
    # recommend_book_index = np.argsort(-similarity_matrix[book_idx][1:11])
    query = book_comms[book_idx] # 获取输入图书的评论
    scores = bm25_model.get_scores(query) # 获取输入图书的BM25分数
    recommend_book_index = np.argsort(-scores)[1:11] # 获取BM25分数排序后的索引
    


    # 输出推荐的图书
    for idx in recommend_book_index:
        # print(f"《{book_names[idx]}》 \t 相似度：{similarity_matrix[book_idx][idx]:.4f}")
        print(f"《{book_names[idx]}》 \t 相似度：{scores[idx]:.4f}")
    
    print("推荐结束")



    
    