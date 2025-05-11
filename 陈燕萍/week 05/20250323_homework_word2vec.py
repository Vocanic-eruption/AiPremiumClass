import jieba

with open("gongyiji.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 用 jieba 分词并保存新语料
with open("gongyiji_cut.txt", "w", encoding="utf-8") as f:
    for line in lines:
        if line.strip():
            words = jieba.lcut(line.strip())
            f.write(" ".join(words) + "\n")

import fasttext

# 训练 skip-gram 模型（word2vec）
model = fasttext.train_unsupervised("gongyiji_cut.txt", model='skipgram', dim=100, ws=5, minCount=1, epoch=10)

# 查看词向量
print(model.get_word_vector("孔乙己"))

# 查看与“孔乙己”最相近的词
print("与‘孔乙己’最相近的词：")
for sim, word in model.get_nearest_neighbors("孔乙己"):
    print(f"{word}：{sim:.4f}")

# 查看“孔乙己”与“掌柜”的相似度（余弦距离）
from numpy import dot
from numpy.linalg import norm

vec1 = model.get_word_vector("孔乙己")
vec2 = model.get_word_vector("掌柜")
cos_sim = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
print(f"‘孔乙己’ 与 ‘掌柜’ 的相似度：{cos_sim:.4f}")

# 保存模型
model.save_model("fasttext_model.bin")

# 重新加载模型
model = fasttext.load_model("fasttext_model.bin")