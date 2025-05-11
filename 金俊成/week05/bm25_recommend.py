from rank_bm25 import BM25Okapi
import csv
import jieba
import os
import heapq
from nltk.corpus import stopwords
import string
stopwords_zh = set(stopwords.words('chinese'))
stopwords_zh.update(set(string.punctuation))
def load_data(filename):
    book_comments = {}
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for item in reader:
                book = item['book']
                comment = item['body']
                if not book or not comment.strip(): 
                    continue
                
                comment_words = jieba.lcut(comment)
                comment_words = [word for word in comment_words if word not in stopwords_zh]
                if not comment_words:
                    continue
                
                book_comments.setdefault(book, [])
                book_comments[book].extend(comment_words)
    except FileNotFoundError:
        print(f"文件未找到：{filename}")
    except Exception as e:
        print(f"加载数据时发生错误：{e}")
    return book_comments

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(script_dir, 'files', 'doubanbook_top250_fixed.txt')

# 加载图书评论数据
book_comments = load_data(data_file_path)

if not book_comments:
    print("未成功加载任何图书评论数据，程序将退出。")
    exit()

# 构建 BM25 模型
# 将corpus转换为分词后的列表形式，每本书的评论作为一个独立文档
corpus = list(book_comments.values())
bm25 = BM25Okapi(corpus)

# 缓存书籍名称列表以提高性能
book_titles = list(book_comments.keys())

def process_query(query):
    """处理用户查询并返回分词结果"""
    query_tokens = jieba.lcut(query)
    return [token for token in query_tokens if token not in stopwords_zh]

def get_top_recommendations(query_tokens, bm25, book_titles, top_n=10):
    """获取推荐结果"""
    scores = bm25.get_scores(query_tokens)
    top_indices = heapq.nlargest(top_n, range(len(scores)), key=lambda x: scores[x])
    return [(book_titles[idx], scores[idx]) for idx in top_indices if scores[idx] > 0]

print("BM25 模型已构建完成，开始查询...")
while True:
    query = input('请输入查询（图书名称，输入 q 或 exit 退出）：').strip().lower()
    if query in ['q', 'exit']:
        print("程序已退出。")
        break

    query_tokens = process_query(query)
    if not query_tokens:
        print("查询内容无效，请重新输入。")
        continue

    recommendations = get_top_recommendations(query_tokens, bm25, book_titles)
    if not recommendations:
        print("未找到与查询相关的图书。")
        continue

    print("推荐结果如下：")
    for title, score in recommendations:
        print(f'《{title}》\t 相似度{score:.4f}')