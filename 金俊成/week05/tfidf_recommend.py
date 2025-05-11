def fix_comments():
    fixed = open('week04\doubanbook_top250_fixed.txt','w',encoding='utf-8')
    introduction_file = open('week04\doubanbook_top250_comments.txt','r',encoding='utf-8')
    introduction_list = introduction_file.readlines()
    for i,line in enumerate(introduction_list):
        if i == 0:
            prev_line = ''
            fixed.write(line)
            continue
        terms = line.split('\t')
        # 当前行书名等于上一行书名
        if terms[0] == prev_line.split('\t')[0]:
            if len(prev_line.split('\t')) == 6:
                fixed.write(prev_line+'\n')
                prev_line = line.strip()
            else:
                prev_line = ''
        else:
            if len(terms) == 6:
                prev_line = line.strip()
            else:
                prev_line += line.strip()
    fixed.close()


import csv
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_data(filename):
    # 图书评论集合
    book_comments = {}
    with open(filename,'r',encoding='utf-8') as f:
        reader = csv.DictReader(f,delimiter='\t')
        for item in reader:
            book = item['book']
            comment = item['body']
            comment_words = jieba.lcut(comment)
            if book == '': continue

            book_comments.setdefault(book,[])
            book_comments[book].extend(comment_words)
    return book_comments

if __name__ == '__main__':
    # 加载停用词列表
    stopwords = [line.strip() for line in open('week04\stopwords.txt','r',encoding='utf-8').readlines()]
    # 加载图书评论数据
    book_comments = load_data('week04\doubanbook_top250_fixed.txt')
    print(len(book_comments))
    # 提取书名和评论文本
    book_names = list(book_comments.keys())
    book_comms = list(book_comments.values())
    # 构建TF-IDF矩阵
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    tfidf_matrix = vectorizer.fit_transform([' '.join(comms) for comms in book_comms])
    # 计算相似度矩阵
    similarity_matrix = cosine_similarity(tfidf_matrix)
    # 输入要推荐的图书
    print(f'当前的图书有：\n{book_names}')
    while True:
        book_name = input('请输入要推荐的图书：')
        if book_name in ['q','exit']:
            break
        book_idx = book_names.index(book_name)
        # 获取与输入图书最相似的图书索引
        recommend_book_index = np.argsort(-similarity_matrix[book_idx])[1:11]
        # 输出推荐结果
        for idx in recommend_book_index:
            print(f'《{book_names[idx]}》\t 相似度{similarity_matrix[book_idx][idx]:.4f}')
