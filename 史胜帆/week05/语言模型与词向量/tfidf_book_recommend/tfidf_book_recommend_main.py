#任务 使用tdidf进行图书评论打分、计算评论余弦相似度并推荐
# 使用数据集 PKU 的 douban_top250_comments.txt

import numpy as np
import torch
import torch.nn as nn
import jieba
import csv #专门读取格式化文本文件的包
from sklearn.feature_extraction.text import TfidfVectorizer  # tfidf包
from sklearn.metrics.pairwise import cosine_similarity  # 计算tfidf矩阵的余弦相似度



#观察数据 进行数据预处理 形成fixed.txt
# #1修复后的文件存盘
# fixed = open('douban_comments_fixed.txt','w',encoding = 'utf-8')
# #2开始修复原始文件
# line = [line for line in open('doubanbook_top250_comments.txt','r',encoding = 'utf-8')]
# for i,line in enumerate(line):
#     if i == 0:
#         fixed.write(line) #第一行属性列写进去
#         pre_line = ''
#         continue
#     pre_terms = pre_line.split('\t')
#     terms = line.split('\t')
    
#     if pre_terms[0] == terms[0]:
#         fixed.write(pre_line + '\n')
#         pre_line = line.strip()
#     else:
#         if(len(terms) == 6):
#             fixed.write(line )
#             pre_line = line.strip()
#         else:
#             pre_line += line.strip()


# fixed.close()
#文件处理好了 就可以把预处理的代码注释掉 下面的工作中只要导入处理后的文件即可

#提取文件
def load_data(filename):
    book_comments = {} # key书名：value这本书的所有评论 汇总在一起

    with open(filename,'r',encoding = 'utf-8') as f:
        reader = csv.DictReader(f,delimiter = '\t') # csv.DictReader() 可以识别文件的标题列 并以标题列为key的字典的形式返回每行的字段 字段默认以,分隔 这里给出以'\t'分隔
        for item in reader:
            book = item['book']
            comment = item['body']
            if comment == None:
                continue
            else:   
                comment_words = jieba.lcut(comment)
            #print(book,comment_words)
        
            #再来一个安全操作 跳过空书名
            if book == '':
                continue
            #一本书的所有评论的收集  所有书的收集入字典 用get方法 当前book的评论values存在则存入 不存在则返回默认值而不是报错
            book_comments[book] = book_comments.get(book,[])
            book_comments[book].extend(comment_words)
            #现在得到了一个book:comments的字典

            #封装成函数 整合数据
    return book_comments






#数据训练



#测试 
# 加载图书信息
if __name__ == '__main__':

    #加载停用词 去除‘的’ ‘了’等无效词
    stop_words = [line.strip() for line in open('stopwords.txt','r',encoding = 'utf-8')]
    #获取图书和评论做成字典
    book_comments = load_data('douban_comments_fixed.txt')
    print(len(book_comments))


    #由于字典底层是hash表 无法把key和values以对应的方式索引 所以先把他们存入list中让他们有顺序
    book_names = []
    book_cmmms = []
    for key,val in book_comments.items():
        book_names.append(key)
        book_cmmms.append(val)



    # 特征提取 构建IFIDF特征矩阵  即计算每个文档的重要性

    vectorizer = TfidfVectorizer(stop_words = stop_words) #在tfidf向量化vector里使用停用词 使用时过滤掉停用词
    tfidf_matrics = vectorizer.fit_transform([' '.join(comments) for comments in book_comments.values()])
    print(tfidf_matrics.shape) # (234,75141)@@
    print(vectorizer.get_feature_names_out()) #打印所有comments中的所有分词的去除停顿词后的所有特征词


    #计算图书之间的余弦相似度
    similarity_matrics = cosine_similarity(tfidf_matrics)
    print(similarity_matrics.shape)  # 每行和自身及其他行计算余弦值 因此是个方阵并且对角线全为1

    #接下来 可以开始推荐了 原理是根据余弦相似度越小 越相似 越推荐
    #print(book_comments.keys())  #若直接打印 输出结果不是从上往下的顺序 因为python的字典是用hash表实现的 会根据键的哈希值来存储和检索 所以输出是乱序
    #那么加list()后就会按照keys的插入顺序输出 这样就是有序的
    
    print([name for name in book_names])
    book_name = input("请输入书名：")
    book_idx = book_names.index(book_name) #获取列表元素索引 .index()

    #由于similarity_matrics是对角阵 所以我们取book_idx所在的行的前x个值 就是最相似的
    # cos值越大 夹角越小 夹角越小越相似 所以加符号
    recommend_books_idx = np.argsort(-similarity_matrics[book_idx])[1:11] #引用类型 不用在赋值 默认升序排序 返回一个索引项列表
    #取最推荐的前10本 因为自身的余弦值是1 所以取前11本才是取到了前十本其他书
    for i in recommend_books_idx:
        print(f'《{book_names[i]}》\t 相似度：{similarity_matrics[book_idx][i]:.4f}')

    
