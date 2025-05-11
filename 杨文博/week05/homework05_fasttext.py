import jieba
import pandas as pd
import fasttext

path_waimai = "./waimai_10k.csv"
df_waimai = pd.read_csv(path_waimai)
# 分词
df_waimai['review'] = df_waimai['review'].apply(lambda x:"".join(jieba.cut(x)))
# 提取评论列并保存为文本文件
df_waimai['review'].to_csv('./reviews.txt', index=False, header=False, sep='\n')

# 使用 fastText 训练词向量，指定使用 skipgram 模型
model = fasttext.train_unsupervised('reviews.txt', model='skipgram')

# 保存训练好的模型
model.save_model('word_vectors.bin')
similar_words = model.get_nearest_neighbors('外卖', k=5)
print(similar_words)

df = pd.read_csv("cooking.stackexchange/cooking.train", sep="\t", header=None)
df.columns = ["label", "text"]
df["label"] = df["label"].apply(lambda x: "__label__" + str(x))
df.to_csv("cooking.train", index=False, header=False, sep=" ")
model2 = fasttext.train_supervised(input="cooking.train", epoch=25, lr=1.0, wordNgrams=2, verbose=2, minCount=1)