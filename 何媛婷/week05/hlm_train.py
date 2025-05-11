import jieba
from gensim.models import FastText

def fetch_fasttext_model():
    """
    获取模型
    """
    return FastText(
        sentences,
        vector_size=100,    # 词向量维度
        window=10,          # 扩大窗口捕捉古典文本的长距离依赖
        min_count=1,        # 章节较短，保留所有词
        workers=4,
        sg=1,               # 使用skip-gram模型（更适合小数据）
        hs=0,               # 使用负采样
        negative=5,         # 负采样数量
        epochs=50           # 增加迭代次数
    )

def fetch_hlmtext():
    with open("./data/hlm_c.txt", "r", encoding="utf-8") as f:
        text = f.read()
    jieba.load_userdict("./data/hlm_dict.txt")
    return [jieba.lcut(text)]

if __name__=='__main__':
    """
    使用自定义的文档文本,通过fasttext训练word2vec训练词向量模型,并计算词汇间的相关度
    """
    # 1. 读取文本
    sentences = fetch_hlmtext()

    # 2. 训练FastText
    model = fetch_fasttext_model()

    # 3. 计算词汇相关度
    print("测试1: 贾雨村 vs 甄士隐 相似度:", model.wv.similarity("贾雨村", "甄士隐"))
    print("测试2: 与 贾宝玉 相关的词:", model.wv.most_similar("贾宝玉", topn=5))

    # 4. 保存
    model.save("./model/hlm_fasttext.model")


