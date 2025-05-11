from rank_bm25 import BM25Okapi
import jieba



corpus_chinese = [
    '当 晨曦 拥抱 这座城',
    '指引着 赶路的 旅人',
    '给世界 留下一抹 温存',
    '幸福 恰好决定 与自己 相认',
    '听 河流 轻声 在哼唱',
    '贪睡的 繁星 也渴望',
]


# 中文文本分词
corpus_chinese = [' '.join(jieba.cut(text)) for text in corpus_chinese]

# 训练bm25模型
bm25 = BM25Okapi(corpus_chinese)

query = '的'
tokenized_query = list(jieba.cut(query))

# 计算文档得分
scores = bm25.get_scores(tokenized_query)
print("各文档BM25得分:", scores)



# 获取最相关的3个文档
top3_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
print("\n最相关的3个文档：")
for idx in top3_indices:
    print(f"[得分 {scores[idx]:.2f}] {corpus_chinese[idx]}")

