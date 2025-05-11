"""
1. 文本清洗，数据预处理，分词
2. 构建数据集
3. 训练测试
"""
import pickle

import pandas as pd
import matplotlib.pyplot as plt

from cut_words import cut_words_jieba, cut_words_sentence_pieces_test


def get_comments(df):
    comments = []
    for vote, cmt in zip(df['Star'].astype(int), df['Comment']):
        if 10 < len(cmt) < 5000:
            if vote in [1, 2]:
                comments.append((cmt, '1'))
            if vote in [4, 5]:
                comments.append((cmt, '0'))
    # cmts = comments[1] + comments[0]
    # cmt_len = [len(x) for x in cmts]
    # plt.hist(cmts, bins=100)
    # plt.show()
    return comments


def build_from_doc(doc):
    vocab = set()
    for line in doc:
        vocab.update(line[0])
    # padding 填充, 解决不同长度文本的对齐， UNK: Unknown， 对未知词汇进行编码
    vocab = ["<PAD>", 'UNK'] + list(vocab)
    w2idx = {word: idx for idx, word in enumerate(vocab)}
    return w2idx


if __name__  == '__main__':
    data_path = '../data/DMSC.csv/DMSC.csv'
    data = pd.read_csv(data_path)
    comments = get_comments(data)
    with open('douban_comments.csv', 'w') as f:
        for comment in comments:
            f.write(comment[0] + '\n')

    for cut_type in ['spm', 'jeiba']:
        if cut_type == 'spm':
            comments_split = cut_words_sentence_pieces_test(comments)
        else:
            comments_split = cut_words_jieba(comments)
        w2idx = build_from_doc(comments_split)
        with open(f'{cut_type}_db_w2idx.pkl', 'wb') as f:
            pickle.dump(w2idx, f)
        with open(f'{cut_type}_db_comments.pkl', 'wb') as f:
            pickle.dump(comments_split, f)



