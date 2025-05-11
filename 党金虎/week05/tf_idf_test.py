from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
corpus_english = [
    'This is the first document.', # 第一个文档
    'This document is the second document.', # 第二个文档
    'And this is the third one.', # 第三个文档
    'Is this the first document?', # 第四个文档
]

# 将文本数据转换为tf-idf特征
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus_english)
# 输出特征名
print(vectorizer.get_feature_names_out())
print(X.shape)


corpus_chinese = [
    '当 晨曦 拥抱 这座城',
    '指引着 赶路的 旅人',
    '给世界 留下一抹 温存',
    '幸福 恰好决定 与自己 相认',
    '听 河流 轻声 在哼唱',
    '贪睡的 繁星 也渴望',
]

# 将文本数据转换为tf-idf特征
vectorizer = TfidfVectorizer()
# 中文文本分词
corpus_chinese = [' '.join(jieba.cut(text)) for text in corpus_chinese]
# 转换为tf-idf特征
X = vectorizer.fit_transform(corpus_chinese)
# 输出特征名
print(vectorizer.get_feature_names_out())
print(X.shape)
# 输出特征矩阵
print(X.toarray())

