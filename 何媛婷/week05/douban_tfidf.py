from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
    实现基于豆瓣top250图书评论的简单推荐系统(TF-IDF算法实现)
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
    # Step 2: 构建TF-IDF矩阵
    
    print('TF-IDF矩阵构建中....')
    tfidf_vectorizer=TfidfVectorizer(stop_words=stopwords)
    tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(bookcomments[bookname]) for bookname in bookslist])
    print('TF-IDF矩阵构建完毕!')

    searchbook = input("请输入图书名称:")
    print(f"本次将根据《{searchbook}》书名进行推荐，正在计算中......\n")


    # ------------------------------------------------------------------------------------------
    # Step 3: 将原始文本转换为TF-IDF特征矩阵

    query_vec = tfidf_vectorizer.transform([' '.join(bookcomments[searchbook])])


    # ------------------------------------------------------------------------------------------
    # Step 4: 计算书籍相似度,找到与目标图书最相似的图书

    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    # 取相似的排名前10的图书
    top_indices = sim_scores.argsort()[-10:][::-1]


    # ------------------------------------------------------------------------------------------
    # Step 5: 输出相似结果
    
    print(f"top_indices为 {top_indices}\n")
    print("推荐结果，为您推荐的图书有:\n")
    for idx in top_indices:
        print(f"《{bookslist[idx]}》，相似度:{sim_scores[idx]} ;\n")


