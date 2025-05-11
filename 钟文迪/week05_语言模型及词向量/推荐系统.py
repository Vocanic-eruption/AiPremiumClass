import csv
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_data(filename):
    # 图书评论信息集合
    book_comments = {}  # {书名：“评论1词 + 评论2词 + ...”}

    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')  # 识别格式文本中标题列
        for item in reader:
            book = item['book']
            comment = item['body']
            comment_words = jieba.lcut(comment)

            if book == '': 
                continue  # 跳过空书名

            # 图书评论集合收集
            book_comments[book] = book_comments.get(book, [])
            book_comments[book].extend(comment_words)
    
    return book_comments

if __name__ == '__main__':

    # 加载停用词
    stop_words = [line.strip() for line in open('./钟文迪/week05_语言模型及词向量/stop_words.txt', 'r', encoding='utf-8')]
    # print(stop_words)

    # 加载图书评论
    book_comments = load_data("./钟文迪/week05_语言模型及词向量/process_douban_comments.txt")
    # print(book_comments)

    # 提取书名和评论文本
    book_names = []
    book_comms = []
    for book, comments in book_comments.items():
        book_names.append(book)
        book_comms.append(comments)

    # 构建TF-IDF特征矩阵
    vectorize = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorize.fit_transform([" ".join(comments) for comments in book_comms])

    # 计算图书之间的余弦相似度
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(similarity_matrix)

    # 输入要推荐的图书名称
    book_list = list(book_comments.keys())
    # print(book_list)
    book_name = input("请输入图书名称：")
    book_index = book_list.index(book_name)

    # 获取与输入图书最相似的图书
    recommend_book_index = np.argsort(-similarity_matrix[book_index])[1:11]

    # 输出推荐的图书
    for index in recommend_book_index:
        print(f"《{book_names[index]}》 \t 相似度：{similarity_matrix[book_index][index]:.4f}")
    
    print("推荐完毕！")