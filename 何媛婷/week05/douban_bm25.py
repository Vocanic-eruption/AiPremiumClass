from rank_bm25 import BM25Okapi
import numpy as np
import jieba
import pandas as pd 

def load_douban_data(doubanbook):
    """ 
    返回豆瓣读书的书名和评论信息
    """
    bookcomments={}
    # 所有评论作为文档集合
    bookbodys=doubanbook['body'].tolist()
    booknames=doubanbook['book'].tolist()
    for bookname,bookbody in zip(booknames,bookbodys):
        bookcomment =jieba.lcut(str(bookbody))
        if bookname == "":continue
        bookcomments[bookname]=bookcomments.get(bookname, list())
        bookcomments[bookname].extend(bookcomment)
    return bookcomments

if __name__=='__main__':
    """
    实现基于豆瓣top250图书评论的简单推荐系统(BM25算法实现)
    """
    # 读取源数据
    doubanbook = pd.read_excel('./data/book_douban.xlsx')

    # ------------------------------------------------------------------------------------------
    # Step 1: 文本预处理（分词+去停用词）

    print('图书评论数据加载中...')
    bookcomments = load_douban_data(doubanbook)
    print("图书评论数据加载完毕!")
    bookslist =list(bookcomments.keys())
    print('图书数量:',len(bookslist))
    # 中文分词
    stopwords = [word.strip()for word in open('./data/stopwords.txt','r',-1,encoding="utf-8")]
    print("中文分词数据加载完毕!")

    # ------------------------------------------------------------------------------------------
    # Step 2: 构建BM25模型

    print('BM25模型构建中....')
    tokenized_corpus = [' '.join(bookcomments[bookname]) for bookname in bookslist]
    bm25 = BM25Okapi(tokenized_corpus)
    print('BM25模型矩阵构建完毕!')
    searchbook = input("请输入图书名称:")
    print(f"本次将根据《{searchbook}》书名进行推荐，正在计算中......\n")


    # ------------------------------------------------------------------------------------------
    # Step 3: 计算书籍相似度,找到与目标图书最相似的图书

    sim_scores = bm25.get_scores(bookcomments[searchbook])
    # 取相似的排名前10的图书
    top_indices = sim_scores.argsort()[-10:][::-1]


    # ------------------------------------------------------------------------------------------
    # Step 4: 输出相似结果
    
    print(f"top_indices为 {top_indices}\n")
    print("推荐结果，为您推荐的图书有:\n")
    for idx in top_indices:
        score = (sim_scores[idx] - min(sim_scores)) / (max(sim_scores) - min(sim_scores))
        print(f"《{bookslist[idx]}》，相似度:{score} ;\n")


