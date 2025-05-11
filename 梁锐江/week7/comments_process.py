import csv
import pickle

import jieba
import matplotlib.pyplot as plt

'''
    数据分析
'''


def plot_doc_len(comments_len):
    plt.hist(comments_len, bins=100)
    plt.show()


def plot_blox_plot(comments_len):
    plt.boxplot(comments_len)
    plt.show()


'''
    数据清理
'''


def data_clean(docs):
    return [doc for doc in docs if len(doc[0]) in range(10, 111)]


if __name__ == '__main__':
    data = []

    with open('comments.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for line in reader:
            vote = int(line['votes'])
            comment_words = jieba.lcut(line['content'])
            if vote in range(0, 5):
                data.append((comment_words, 1 if vote in [0, 1, 2] else 0 if vote in [4, 5] else 2))

    # print(data[-1])
    comments_len = [len(c) for c, v in data]
    # plot_doc_len(comments_len)
    # plot_blox_plot(comments_len)

    docs_fixed = data_clean(data)
    fixed_data_len = [len(d) for d, v in docs_fixed]
    plot_doc_len(fixed_data_len)
    plot_blox_plot(fixed_data_len)

    with open('fixed_comments.pkl', 'wb') as fixed_f:
        pickle.dump(docs_fixed, fixed_f)
