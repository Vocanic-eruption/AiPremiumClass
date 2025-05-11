import torch
import numpy as np
import csv
from tqdm import tqdm
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_tsv_data(file_name):
    book_comments = {}
    with open(file_name, 'r', encoding='utf-8') as f:
        #reader = csv.reader(f, delimiter='\t')   #reader是迭代器。每次迭代会返回文件中的一行数据，并解析为一个列表list，列表中的每个元素是一个字段。
        #与 csv.reader 不同，csv.DictReader 会将每一行数据解析为一个有序字典
        reader = csv.DictReader(f, delimiter='\t')   # 读取tsv文件，默认分隔符为 ，这里用的分隔符delimiter是 \t(制表符)
        for item in reader:
            book_name = item['book']
            comments = item['body']
            coments_words = jieba.lcut(comments)
            
            if book_name =='': continue
            
            book_comments[book_name] = book_comments.get(book_name, [])  # 查看字典中是否存在这本书，不在的话就返回空列表list,一本书对应多个评论
            book_comments[book_name].extend(coments_words)               # extend() 方法用于在列表末尾一次性追加分词后的结果
                                    # append：将一个元素添加到列表的末尾(元素、元组或列表)，但extend是将可迭代的对象追加到列表的末尾
                                    # .append([5,6])：[1, 2, 3, 4, [5, 6]]  .extend([5,6])：[1, 2, 3, 4, 5, 6]
        return book_comments                                             # 最后将书名对应分词后的值的字典返回
            

if __name__ == '__main__':
    filename = 'doubanbook_fixed.txt'
    #使用strip()方法去掉word的首尾空白字符（包括空格、换行符等）。这样可以确保列表中的每个停用词都是干净的字符串。
    stopwords = [ word.strip() for word in open('stopwords.txt', 'r', encoding='utf-8')] #open返回可迭代的对象，然后用word去除首尾空白字符。
    print('加载图书评论数据...')
    book_comments = load_tsv_data(filename)
    print('图书评论数据加载完成')
    book_list = list(book_comments.keys())  #获取处理后的图书名称
    print(f'图书数量为:{len(book_list)}')
    
    print("构建TF-IDF矩阵...")
    vectorizer = TfidfVectorizer(stop_words=stopwords)  # 去除停用词
    #对文本数据进行拟合（fit）转换（transform）拟合（fit）：计算词汇表和 IDF 值。转换（transform）：将文本数据转换为 TF-IDF 特征矩阵。
    tfidf_matrix = vectorizer.fit_transform([''.join(book_comments[bookname]) for bookname in book_list])# 生成TF-IDF矩阵
    print("TF-IDF矩阵构建完成")       # 
    
    #计算tfidf矩阵的余弦相似度
    print('计算余弦相似度...')
    similar_matrix = cosine_similarity(tfidf_matrix) #计算余弦相似度矩阵 shape:(234,234) 234本书
    print('余弦相似度计算完成')
    
    print('找到相似的图书')
    input_name = input('请输入书名:')
    book_index = book_list.index(input_name) # 获取输入的图书在图书列表中的索引
    recommend_book_idies = np.argsort(-similar_matrix[book_index])[:11][1:]      # 获取相似度最高的图书的索引
    
    print('推荐的图书有...')
    for index in recommend_book_idies:
        print(book_list[index])
        print(f'相似度:{similar_matrix[book_index][index]}')

    
# str = '我在家吃苹果'
# print(f'分词结果为{list(jieba.lcut(str))}')    
# print(f'长度为{len(list(jieba.lcut(str)))}')
# print(f'单个词为{jieba.lcut(str)}')
    