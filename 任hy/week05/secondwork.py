import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. 训练模型
model = fasttext.train_unsupervised(
    'custom_text.txt',
    model='skipgram',
    dim=100,
    ws=5,
    minCount=1,
    epoch=50,
    lr=0.05
)
model.save_model('custom_model.bin')

# 2. 定义相似度计算函数
def word_similarity(word1, word2):
    vec1 = model.get_word_vector(word1).reshape(1, -1)
    vec2 = model.get_word_vector(word2).reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

def most_similar_words(word, topn=5):
    neighbors = model.get_nearest_neighbors(word, k=topn)
    return [(word, sim) for sim, word in neighbors]

# 3. 计算和展示结果
words_to_test = [("自然语言处理", "人工智能"), 
                ("词向量", "Word2Vec"), 
                ("深度学习", "Transformer")]

print("\n词汇相似度计算:")
for word1, word2 in words_to_test:
    sim = word_similarity(word1, word2)
    print(f"'{word1}'和'{word2}'的相似度: {sim:.4f}")

print("\n查找相似词示例:")
target_words = ["NLP", "模型", "方法"]
for word in target_words:
    print(f"\n与'{word}'最相似的词:")
    similar_words = most_similar_words(word)
    for w, s in similar_words:
        print(f"  {w}: {s:.4f}")
