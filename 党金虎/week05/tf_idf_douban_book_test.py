from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import csv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



def load_data(filename):
    book_comments = {}
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t') # 使用csv模块读取数据
        for item in reader:
            # book	id	star	time	likenum	body
            book_name = item[0]
            comments = item[5]
            # cut和lcut区别在于cut返回的是一个生成器,而lcut返回的是一个列表
            comment_words = jieba.lcut(comments)
            if book_name == '' or book_name == 'book': continue
            # 以书名为键,评论为值
            book_comments[book_name] = book_comments.get(book_name, list()) 
            # 将评论添加到对应书名的评论列表中
            book_comments[book_name].extend(comment_words)
    return book_comments


if __name__ == '__main__':
    # 1. 加载图书评论数据
    file_name = './党金虎/week05/douban_test_data/doubanbook_top250_comments_fixed.txt'
    book_comments    = load_data(file_name)
    print('图书数量:', len(book_comments))
    book_list = list(book_comments.keys())
    print('书名:', book_list[:20])
    # 2. 采用 stopwords 作用是去除一些无意义的词语
    stopwords = [line.strip() for line in open('./党金虎/week05/douban_test_data/stopwords.txt', 'r', encoding='utf-8')]
    print('停用词数量:', len(stopwords))

    # 3. 构建tf-idf特征或
    vectorizer =  TfidfVectorizer(stop_words=stopwords) # 去除停用词
    # 将评论列表转换为字符串
    tfidf_matrix =  vectorizer.fit_transform([' '.join(book_comments[book_name]) for book_name in book_list])
    print('特征矩阵维度:', tfidf_matrix.shape)

    # 4. 计算两个向量的余弦相似度
    similatiy_matrix = cosine_similarity(tfidf_matrix)
    print('相似度矩阵维度:', similatiy_matrix.shape)

    # 5. 查找与指定书籍最相似的书籍
    boot_name = input('请输入书名:')
    book_idx = book_list.index(boot_name)
    # 获取与指定书籍最相似的书籍
    most_similar_books = np.argsort(similatiy_matrix[book_idx])[::-1] #::-1表示逆序,从大到小 
    print('与', boot_name, '最相似的书籍:')
    for idx in most_similar_books[1:6]:
        print(book_list[idx], similatiy_matrix[book_idx][idx])






















    # test 列表取值方式有哪些
    list = [1,2,3,4,5]
    # 1. list[0] 通过索引取值 = 1
    # 2. list[1:5] 切片取值 = [2,3,4,5]
    # 3. list[::-1] 逆序取值 = [5,4,3,2,1]
    # 4. list[::2] 步长取值 = [1,3,5]
    # 5. list[1:] 从指定位置开始取值 = [2,3,4,5]
    # 6. list[:5] 从开始位置取值 = [1,2,3,4,5]
    # 7. list[-1] 取最后一个值 = 5
    # 8. list[-2] 取倒数第二个值 = 4
    # 9. list[-5:] 取倒数5个值 = [1,2,3,4,5]
    # 10. list[:-5] 取除了最后5个值的其他值 = []
    # 11. list[1:-1] 取除了第一个和最后一个值的其他值 = [2,3,4]
    # 12. list[::] 取所有值 = [1,2,3,4,5]
    # 13. list[1:5:2] 取指定步长的值 = [2,4]
