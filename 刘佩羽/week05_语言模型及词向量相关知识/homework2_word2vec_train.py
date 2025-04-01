# 需要空格分词的文本文件才能训练模型，所以需要先分词
import fasttext
import jieba


# 分词
def cut_words(file_path, file_name):
    with open(file_path, mode='r', encoding='utf-8') as f:
        text = f.read()

    # 分词，并保存到一个文件中
    with open(file_name, mode='w', encoding='utf-8') as f:
        # f.write(' '.join(jieba.cut(text.replace(' ', '').replace('\n', ''))))
        f.write(' '.join(jieba.cut(text)))
    print('分词完成')


if __name__ == '__main__':
    cut_words('./《红楼梦》完整版.txt', './hongloumeng.txt')
    # 询问用户使用的word2vec模型的类型，给出选项
    model_type = input('请选择word2vec模型的类型，1表示skipgram，2表示cbow：')
    if model_type == '1':
        model_type = 'skipgram'
    elif model_type == '2':
        model_type = 'cbow'
    else:
        print('输入错误，请重新输入')
        exit()

    # 训练word2vec模型,使用cbow模型
    model = fasttext.train_unsupervised('./hongloumeng.txt', model=model_type,
                                        dim=100, epoch=10, lr=0.1, wordNgrams=2,
                                        loss='ns', bucket=200000, thread=4,
                                        minCount=1)
    # 保存模型
    model.save_model('hongloumeng.bin')

    # 加载模型
    model = fasttext.load_model('hongloumeng.bin')
    # 查看模型信息
    # print(model.words)
    # 查看分词的数量
    # print('文档中分词的数量为', len(model.words))
    # 获取词向量
    # print('获取词向量', '\n', model.get_word_vector('贾宝玉'))
    # 查看词向量维度
    print('词向量的维度为', model.get_dimension())
    # 计算词汇间的相关度
    print('计算词汇间的相关度', '\n', model.get_nearest_neighbors('空空'))


