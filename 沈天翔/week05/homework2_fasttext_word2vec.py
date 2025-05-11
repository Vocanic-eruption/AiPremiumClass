import fasttext
import jieba
import os

book_name = 'san_guo.txt'
sparse_book_name = 'san_guo_sparse.txt'

# 读取文件内容
if not os.path.exists(sparse_book_name):
    with open(book_name, 'r', encoding='utf-8') as f:
        book_text = f.read()
        # print(book_text[:100])
        book_words = jieba.lcut(book_text)
        # print(book_words[:100])
        with open(sparse_book_name, 'w', encoding='utf-8') as f:
            f.write(' '.join(book_words))

# 训练语言模型
# model = fasttext.train_unsupervised(sparse_book_name, model='cbow', minCount=10)
model = fasttext.train_unsupervised(sparse_book_name, model='skipgram', minCount=10)

neibor_words = model.get_nearest_neighbors('袁绍', k = 10)
# print(neibor_words)

# 词汇间类比
analogy_words = model.get_analogies('刘备', '曹操', '孙权', k=10) # 刘备 - 曹操 + 孙权
# print(analogy_words)

# 模型保存
model.save_model('san_guo.bin')

# 加载模型
model = fasttext.load_model('san_guo.bin')

# 预测词向量
word_vector = model.get_word_vector('袁绍')
# print(word_vector)


