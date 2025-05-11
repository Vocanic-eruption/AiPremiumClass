#实现基于豆瓣top250图书评论的简单推荐系统（TF-IDF及BM25两种算法实现）

#1、处理文档为可用
# reset_text = open("reset_book_text.txt","w",encoding="utf-8")
# lines = [line for line in open("doubanbook_top250_comments.txt","r",encoding="utf-8")]
# print(len(lines)) #99665

# for index,line in enumerate(lines):
#     if index ==0:
#         reset_text.write(line) #写入第一行
#         prev_line = '' #将上一行的内容置为空
#         continue
#     terms = line.split("\t") #按照制表符进行分割
    
#     #如果上一行和当前行是同一部图书
#     if terms[0] == prev_line.split("\t")[0]: 
#         if len(prev_line.split("\t")) == 6: #上一行是评论
#             reset_text.write(prev_line+'\n')
#         else:
#             prev_line=''
#     else:
#         if len(terms) == 6: #当前行是评论
#            prev_line = line.strip()
#         else:
#             prev_line = prev_line + line.strip()
# reset_text.close()
            
#2、计算TF-IDF并通过余弦相似度给出推荐列表
import jieba #分词
import csv #读取csv文件
from sklearn.feature_extraction.text import TfidfVectorizer #计算TF-IDF
from sklearn.metrics.pairwise import cosine_similarity #计算余弦相似度
import numpy as np


def load_data():
    """读取数据"""
    comments_obj = {}  # {书名：评论xxx}
    with open("reset_book_comments.txt", 'r',encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            book = row['book']
            comment = row['body']
            comment_words = jieba.lcut(comment)
            if book == '': continue #跳过空行
            comments_obj[book] = comments_obj.get(book, [])
            comments_obj[book].extend(" ".join(comment_words))
    print(comments_obj)
    return comments_obj

book_comments = load_data()
if __name__ == '__main__':

    stop_words_arr = [line.strip() for line in open("stopwords.txt", "r",encoding="utf-8")] #停用词
    book_comments = load_data()
    book_names = []
    book_comms = []
    for book, comments in book_comments.items():
        book_names.append(book)
        book_comms.append(comments)
        
        
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([' '.join(comms) for comms in book_comms])
    similarity_res = cosine_similarity(matrix)
    tfidf_res = similarity_res
    
    
    # 推荐测试
    book_list = list(book_comments.keys())
    book_name = input("请输入图书名称：")
    book_idx = book_names.index(book_name)  # 获取图书索引
    
    recommend_book_index = np.argsort(-tfidf_res[book_idx])[1:10]
    # 输出推荐的图书
    for idx in recommend_book_index:
        print(f"《{book_names[idx]}》 \t 相似度：{tfidf_res[book_idx][idx]:.4f}")