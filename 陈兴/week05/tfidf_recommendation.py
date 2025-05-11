from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import csv

# 读取评论数据
file_path = '/Users/chenxing/AI/AiPremiumClass/陈兴/week05/doubanbook_top250_comments_single_line.txt'

def load_data(file_path):
    book_comments = {} # {书名: [评论1, 评论2, ...]}
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for item in reader:
            book = item['book']
            comment = item['body']
            if book == '' or comment == '':
                continue
            
            # 使用jieba进行中文分词
            segmented_comment = ' '.join(jieba.lcut(comment))
            if book not in book_comments:
                book_comments[book] = []
                
            book_comments[book].append(segmented_comment)
    return book_comments

if __name__ == '__main__':

 # 提取书名和评论
    book_comments = load_data(file_path)    
    books = list(book_comments.keys())
    comments = list(book_comments.values())

    # 计算TF-IDF矩阵
    # 加载停用词表
    # 使用绝对路径读取停用词表
    with open('/Users/chenxing/AI/AiPremiumClass/陈兴/week05/stopwords.txt', 'r', encoding='utf-8') as f:
        stop_words = [line.strip() for line in f]
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(comments)

    # 计算余弦相似度
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 推荐函数
    def get_recommendations(title, cosine_sim=cosine_sim):
        # 获取书名索引
        idx = books.index(title)
        
        # 获取相似度分数
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # 按相似度排序
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # 获取最相似的10本书
        sim_scores = sim_scores[1:11]
        
        # 获取书名和相似度
        return [(books[i[0]], i[1]) for i in sim_scores]

    # 示例：获取与"天才在左 疯子在右"相似的书籍
    recommended_books = get_recommendations("天才在左 疯子在右")
    print("推荐书籍列表:")
    for book, similarity in recommended_books:
        print(f"{book} - 相似度: {similarity:.4f}")
