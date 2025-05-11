# 使用bm25 计算词的重要性 生成bm25矩阵 计算相似性 基于图书评论 推荐书籍
import numpy as np
import csv
import bm25_code  # type: ignore
import jieba
from sklearn.metrics.pairwise import cosine_similarity

# fixed = open('douban_top250_comments_fixed.txt','w',encoding = 'utf-8') 
# lines = [line for line in open('doubanbook_top250_comments.txt','r',encoding = 'utf-8')]
# for i,line in enumerate(lines):
#     if i == 0:
#         fixed.write(line)
#         pre_line = ''
#         continue
#     terms = line.split('\t')
#     pre_terms = pre_line.split('\t')

#     if pre_terms[0] == terms[0]:
#         fixed.write(pre_line + '\n')
#         pre_line = line.strip()
#     else:
#         if len(terms) == 6:
#             fixed.write(line.strip())
#             pre_line = line.strip()
#         else:
#             pre_line += line.strip()
    
# fixed.close()
if __name__ == '__main__':

    with open('douban_top250_comments_fixed.txt','r',encoding = 'utf-8') as f:

        reader = csv.DictReader(f,delimiter = '\t')
        book_comments = {}
        book_names = []
        book_commts = []
        for item in reader:
            if item['book'] == '':continue
            if item['body'] == None:
                continue
            else:
                book_names.append(item['book'])

                comments_words = (jieba.lcut(item['body'])) #jieba分词后返回一个list
                book_commts.extend(comments_words)
                book_comments[item['book']] = book_comments.get(item['book'],[])
                book_comments[item['book']].extend(comments_words)
    
    stop_words = [line for line in open('stopwords.txt','r',encoding = 'utf-8')]
    # 疑问 如何在生成bm25矩阵前 手动去除comments中的停用词
    bm25_metrics = bm25_code.bm25(book_comments,k = 1.5,b = 0.75)

    print(bm25_metrics.shape)

    similarity = cosine_similarity(bm25_metrics)
    book_names = list(set(book_names))
    print(book_names)
    book = input("请输入一本书的名字:")
    book_idx = book_names.index(book)
    recommend_books_idx = np.argsort(-similarity[book_idx])[1:11]
    for i in recommend_books_idx:
        print(f'《{book_names[i]}》,相似度：{similarity[book_idx][i]:.4f}')
    
