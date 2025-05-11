import csv
import jieba 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def clean_comments():
    # 修复后的内容存储到文件
    fixed = open("/Users/chenxing/AI/AiPremiumClass/陈兴/week05/douban_comments_fixed.txt", "w", encoding="utf-8")
    # 修复前的内容文件
    lines = [line for line in open("/Users/chenxing/AI/AiPremiumClass/陈兴/week05/doubanbook_top250_comments.txt", "r", encoding="utf-8")]
    for i, line in enumerate(lines):
        # 保存标题列
        if i == 0:
            fixed.write(line)
            prev_line = ''  # 上一行的书名置为空
            continue

        # 提取书名和评论文本
        terms = line.split("\t")
        # 当前行的书名 == 上一行的书名
        if terms[0] == prev_line.split("\t")[0]:
            if len(prev_line.split("\t")) == 6:  # 上一行是评论
                # 保存上一行记录
                fixed.write(prev_line + '\n')
                prev_line = line.strip()  # 保存当前行
            else:
                prev_line = ""
        else:
            if len(terms) == 6:  # 新书评论
                # fixed.write(line)
                prev_line = line.strip()  # 保存当前行
            else:
                prev_line += line.strip()  # 合并当前行和上一行
    fixed.close()

def load_data(file_path):
    book_comments = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for item in reader:
            book = item['book']
            comment = item['body']
            if book == '' or comment == '':
                continue
            book_comments[book] = book_comments.get(book, []) 
            book_comments[book].append(comment)
    return book_comments

if __name__ == "__main__":
    clean_comments()
    # 加载停用词列表
    stop_words = [line.strip() for line in open("/Users/chenxing/AI/AiPremiumClass/陈兴/week05/stopwords.txt", "r", encoding="utf-8")]

    # 加载图书评论信息
    book_comments = load_data("/Users/chenxing/AI/AiPremiumClass/陈兴/week05/douban_comments_fixed.txt")
    print(len(book_comments))

    # 提取书名和评论文本
    book_names = []
    book_comms = []
    for book, comments in book_comments.items():
        book_names.append(book)
        book_comms.append(comments)

    # 构建TF-IDF特征矩阵
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform([' '.join(comms) for comms in book_comms])

    # 计算图书之间的余弦相似度
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # 输入要推荐的图书名称
    book_list = list(book_comments.keys())
    # print(book_list)
    # book_name = input("请输入图书名称: ")
    book_idx = book_names.index("天才在左 疯子在右")  # 获取图书索引

    # 获取与输入书籍最相似的图书
    recommend_book_index = np.argsort(-similarity_matrix[book_idx])[1:11]

    # 输出推荐书籍
    for idx in recommend_book_index:
        print(f"{book_names[idx]} - {similarity_matrix[book_idx][idx]:.4f} ")