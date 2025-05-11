# encoding: utf-8
# @File  : BookRecommend.py
# @Author: GUIFEI
# @Desc : 图书推荐
# @Date  :  2025/04/02
import json
import TFIDFTrain
import BM25


if __name__ == '__main__':
    # 加载停用词列表
    stop_words = [line.strip() for line in open("../dataset/stopwords/stopwords.txt",'r', encoding="utf-8")]
    # 读取书评分词之后的字点数据
    with open('../dataset/douban/boot_json.json', 'r', encoding='utf-8') as f:
        book_comments = json.load(f)
    # 输入需要推荐的书名
    book_name = input("请输入图书名称： ")
    # 使用TF-IDF算法进行图书推荐
    print("------------------TF-IDF算法图书推荐结果-------------------")
    TFIDFTrain.tfIdf(book_name, book_comments, stop_words)
    print("------------------ BM25算法图书推荐结果 -------------------")
    # 使用BM25算法进行推荐
    BM25.bm25(comments=book_comments, book_name=book_name, stop_words=stop_words)