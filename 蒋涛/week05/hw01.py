

"""
1. 实现基于豆瓣top250图书评论的简单推荐系统（TF-IDF及BM25两种算法实现）

"""
import csv
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def fix_comments(old_file, new_file):
    """
    修复豆瓣图书评论文件，将分割的评论合并到同一行。

    :param old_file: 原始评论文件的路径
    :param new_file: 修复后评论文件的路径
    """
    # 修复后的内容文件存盘文件
    fixed = open(new_file, "w", encoding="utf-8")

    # 修复前内容文件
    lines = [line for line in open(old_file, "r", encoding="utf-8")]
    print(f"修复前文件行数--{len(lines)}--")

    # 逐行读取
    for i, line in enumerate(lines):
        # 保存标题列
        if i == 0:
            # 写入标题行
            fixed.write(line)
            # 上一行的书名置为空
            prev_line = ''
            continue

        # 提取书名和评论文本
        terms = line.split("\t")

        # 当前行的书名 == 上一行的书名，说明当前行是上一行的评论，需要合并
        if terms[0] == prev_line.split("\t")[0]:
            if len(prev_line.split("\t")) == 6:  # 上一行评论完整
                # 保存上一行记录
                fixed.write(prev_line + "\n")
                # 保存当前行
                prev_line = line.strip()
            else:
                # 清空上一行内容
                prev_line = ""
        else:
            if len(terms) == 6:  # 新书评论
                # 保存当前行
                prev_line = line.strip()
            else:  # 旧书评论
                # 合并当前行和上一行的评论
                prev_line += line.strip()

    # 关闭文件
    fixed.close()
    print("修复完成，文件关闭")

    # 读取修复后的文件
    fixeds = [line for line in open(new_file, "r", encoding="utf-8")]
    print(f"修复后文件行数--{len(fixeds)}--")

def load_data(file_path):
    """
    从指定文件中加载图书评论数据，并将评论分词后按图书名称分组存储。

    :param file_path: 包含图书评论数据的文件路径
    :return: 一个字典，键为图书名称，值为该图书评论分词后的列表
    """
    # 用于存储每本书的评论分词列表
    book_comments = {}
    # 打开文件
    with open(file_path, 'r', encoding='utf-8') as file:
        # 创建一个字典读取器，使用制表符作为分隔符
        reader = csv.DictReader(file, delimiter='\t')
        # 逐行读取文件
        for row in reader:
            # 获取图书名称
            book = row['book']
            # 获取评论内容
            comment = row['body']
            # 使用jieba进行分词
            commet_words = jieba.lcut(comment)
            # 如果图书名称为空，则跳过当前行
            if book == '': continue
            # 获取当前图书的评论分词列表，如果不存在则初始化为空列表
            book_comments[book] = book_comments.get(book, [])
            # 将当前评论的分词添加到对应图书的评论分词列表中
            book_comments[book].extend(commet_words)

    return book_comments

def get_book_index(book_name, book_names):
    """
    获取图书名称在图书名称列表中的索引，如果图书不存在则提示用户重新输入。

    :param book_name: 用户输入的图书名称
    :param book_names: 图书名称列表
    :return: 图书名称在列表中的索引
    """
    while True:
        if book_name in book_names:
            return book_names.index(book_name)
        else:
            book_name = input("输入的图书名称不存在，请重新输入：")


def run_tfidf(stop_words_file, comments_fixed):
    """
    运行基于TF-IDF的图书推荐系统。

    :param stop_words_file: 停用词文件的路径
    :param comments_fixed: 包含图书评论数据的文件路径
    """
    # 从停用词文件中读取停用词，并去除每行首尾的空白字符
    stop_words = [line.strip() for line in open(stop_words_file, 'r', encoding='utf-8')]
    
    # 调用 load_data 函数加载图书评论数据
    book_comments = load_data(comments_fixed)
    # 打印图书的数量
    print(len(book_comments))
    
    # 存储图书名称的列表
    book_names = []
    # 存储每本图书评论分词列表的列表
    book_comms = []
    # 遍历 book_comments 字典，将图书名称和评论分词列表分别存储到对应的列表中
    for book, comms in book_comments.items():
        book_names.append(book)
        book_comms.append(comms)

    # 创建一个 TfidfVectorizer 对象，指定停用词列表
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    # 将评论分词列表转换为字符串，并用空格连接，然后计算TF-IDF矩阵
    tfidf_matrix = vectorizer.fit_transform([' '.join(comm) for comm in book_comms])
    
    # 计算TF-IDF矩阵中各文档之间的余弦相似度矩阵
    similarity_matrix = cosine_similarity(tfidf_matrix)
    # print(similarity_matrix.shape)
    
    # 提示用户输入图书名称
    book_name = input("请输入图书名称：")
    # 获取用户输入图书名称在 book_names 列表中的索引
    book_idx = get_book_index(book_name, book_names)
    
    # 对相似度矩阵中与输入图书对应的行进行降序排序，取前10个非自身的图书索引
    recommend_book_index = np.argsort(-similarity_matrix[book_idx])[1:11]
    
    # 遍历推荐图书的索引，打印推荐图书的名称和相似度
    for i in recommend_book_index:
        print(f"《{book_names[i]}》 \t 相似度：{similarity_matrix[book_idx][i]:.4f}")
    
    print()

if __name__ == "__main__":
    # file1 = "/Users/jiangtao/Downloads/dataverse_files/doubanbook_top250_comments.txt"
    # file2 = "/Users/jiangtao/Downloads/dataverse_files/douban_comments_fixed.txt"
    # fix_comments(file1, file2)
    
    stop_words_file = "/Users/jiangtao/Downloads/dataverse_files/stopwords.txt"
    comments_fixed = "/Users/jiangtao/Downloads/dataverse_files/douban_comments_fixed.txt"
    run_tfidf(stop_words_file, comments_fixed)