import csv
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#"douban_comments_fixed.txt"
def load_data(feilname):
  book_comments = {}
  with open(feilname,'r',encoding='utf-8') as f:
      reader = csv.DictReader(f,delimiter='\t') # 识别格式文本中的标题列
      for row in reader:
          book,comments = row['book'],row['body']
          comments_words = jieba.lcut(comments)
          if book =='' : continue # 跳过空格
          book_comments[book] = book_comments.get(book,[])
          book_comments[book].extend(comments_words)
  return book_comments

def bm25(comments,k=2.0,b=0.8):
    # 计算文档总数
    N = len(comments)
     # 初试文档长度列表和词频字典
    doc_lengths = []
    word_doc_freq= {}
    doc_term_dict = [{} for _ in range(N)]# 计算文档长度和词频
    for i,comment in enumerate(comments):
    # 记录文档长度
        doc_lengths.append(len(comment))
        unique_words =  set()
        for word in comment:
            # 统计词频
            doc_term_dict[i][word] = doc_term_dict[i].get(word,0) + 1
            unique_words.add(word)
        for word in unique_words:
            word_doc_freq[word] = word_doc_freq.get(word,0) + 1
  
     # 计算每个单词的平均文档长度
    avg_doc_len = sum(doc_lengths) / N
  
    # 构建词汇表
    vocabulary = list(word_doc_freq.keys())
    word_index = {word: idx  for idx,word in enumerate(vocabulary) }
     # 构建文档 --词汇矩阵
    doc_term_matrix = np.zeros((N,len(vocabulary)))
    for i in range(N):
        for word,freq in doc_term_dict[i].items():
            idx = word_index[word]
            if idx is not None:
                doc_term_matrix[i,idx] = freq
    # 计算IDF 值
    idf_numeraor = N - np.array([word_doc_freq[word] for word in vocabulary]) + 0.5
    idf_denominator = np.array([word_doc_freq[word] for word in vocabulary]) + 0.5
    idf = np.log(idf_numeraor / idf_denominator)
    idf[idf_numeraor<=0] = 0 # 避免出现 nan值
    # 计算BM25值
    doc_lengths = np.array(doc_lengths)
    bm25_matrix = np.zeros((N,len(vocabulary)))
    for i in range(N):
        tf = doc_term_matrix[i]
        bm25 = idf * (tf * (k+1)) / (tf + k *(1- b +b * doc_lengths[i] / avg_doc_len))
        bm25_matrix[i] = bm25
    # 根据原始评论顺序重新排列 bm25值
    final_bm25_matrix = []
    for i,comment in enumerate(comments):
        bm25_comment =[]
        for word in comment:
            idx=word_index.get(word)
            if idx is not None:
                bm25_comment.append(bm25_matrix[i][idx])
        final_bm25_matrix.append(bm25_comment)
    # 找到最长的子列表长度
    max_length = max(len(row) for row in final_bm25_matrix)
    # 将所有子列表填充到相同的长度
    padded_martrix = [row + [0] * (max_length - len(row)) for row in final_bm25_matrix]
    # 转换为 numpy 数组
    final_bm25_matrix = np.array(padded_martrix)
    return final_bm25_matrix

    
  
  
  
  
  
if __name__ == '__main__':
  # 加载停用词列表
  stop_words = [ line.strip() for line in open('stopwords.txt','r',encoding='utf-8') ]
  #加载评论信息
  book_comments = load_data('douban_comments_fixed.txt')
  
  # 提取书名和评论
  book_names = []
  book_comms = []
  for book,comments in book_comments.items():
    book_names.append(book)
    book_comms.append(comments)

  bm25_matrix =  bm25(book_comms)
 
  #构建TF-IDF矩阵
  vectorize = TfidfVectorizer(stop_words=stop_words)
  tfidf_matrix = vectorize.fit_transform([' '.join(comms) for comms in book_comms])
  # 计算余弦相似度
  similarity_mtrix = cosine_similarity(tfidf_matrix)
  # 计算BM25相似度
  similarity_mtrix2 = cosine_similarity(bm25_matrix)
  print(similarity_mtrix.shape)
  print(similarity_mtrix2.shape)
  # 输入要推荐的图书名称
  print(book_names)
  book_name = input('请输入要推荐的图书名称：')
  book_idx = book_names.index(book_name) # 获取图书索引
  
  # 获取与输入图书最相似的图书
  recommend_book_list = np.argsort( -similarity_mtrix[book_idx])[1:11]
  # 使用 bm25 算法获取与输入图书最相似的图书
  recommend_book_list2 = np.argsort(-similarity_mtrix2[book_idx])[1:11]
  print(recommend_book_list2)
  for i in recommend_book_list:
    print(f"《{book_names[i]}》 \t 相似度：{similarity_mtrix[book_idx][i]:.4f}")
  for i in recommend_book_list2:
    print(f"bm25->《{book_names[i]}》 \t 相似度：{similarity_mtrix2[book_idx][i]:.4f}")