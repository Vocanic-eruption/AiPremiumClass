"""
    n-gram本质是通过n个词的排列组合，其中在语料库中 count(n的词) / count(n-1的词) 出现的概率预测，当输入n-1的词时候，出现n的概率
    举例 我喜欢这本书因为它非常有趣
    “我 喜欢” 出现了 1 次
    “喜欢 这” 出现了 1 次
    “这 本书” 出现了 1 次
    “书 因为” 出现了 1 次
    “因为 它” 出现了 1 次
    “它 非常” 出现了 1 次
    “非常 有趣” 出现了 1 次
    P(喜欢∣我)= Count("我 喜欢") / Count("我")
    得出当输入我时，输出喜欢这个词的概率
    优点： 计算简单快速，不需要复杂的参数学习过程
    缺点： 数据稀疏问题：对于未见过的词组合，无法给出合理的估计,长距离依赖性差: 只能考虑固定长度的上下文窗口内的关系，难以补抓长距离依赖

    NNLM模型
    优点：能够学习词的分布式表示（词向量），捕捉词之间的语义关系。具有更好的表达能力，能够捕捉比固定长度上下文窗口更多的信息。
    缺点：计算复杂度较高，尤其是在处理大规模数据集时。需要大量的计算资源进行训练，且训练时间较长。

    1.嵌入层
    首先，将句子中的每个词转换为其对应的词向量。假设词汇表大小为 5，词向量维度为 3，则嵌入矩阵

    词向量矩阵：
    ["我"：[0.1, 0.2, 0.3],
     “喜欢”: [0.4, 0.5, 0.6],
     "这本书": [0.7, 0.8, 0.9],
     "非常":[1.0, 1.1, 1.2],
     "有趣":[1.3, 1.4, 1.5]]

    2.拼接与隐藏层
    输入“我喜欢”，拼接后的向量x为
    x=[0.1,0.2,0.3,0.4,0.5,0.6]
    接下来，将 x 输入到隐藏层进行非线性变换：h = tanh(Wx + b)

    3.输出层
    将隐藏层的输出 h 输入到输出层，计算词汇表中每个词的概率分布
    P(Wt) = softmax(Uh + b')
    权重U 和 b' 是输出层的权重矩阵和偏置项
    经过softmax
    P(Wt) = [0.05,0.05,0.85,0.03,0.02]


"""
import fasttext
import jieba

"""
    无监督学习
            无监督学习与有监督学习相反，它处理的是没有标记的数据，即只提供输入特征而不提供对应的输出标签。
            无监督学习的目标是从数据中发现潜在的结构或模式，比如聚类、关联规则等。
            无监督学习通常用于探索性的数据分析，以理解数据的分布或特征之间的关系。常见的无监督学习任务包括聚类分析（如客户细分）、降维（如主成分分析）和关联规则挖掘。
"""


def train_unsupervised():
    model = fasttext.train_unsupervised('ai.txt', model='skipgram', minCount=1)

    print('词汇长度: ', len(model.words))

    # print(model.get_word_vector("人工智能"))

    # print(model.get_nearest_neighbors("人工智能"))

    print(model.get_analogies("人工智能", "ai", "Python"))

"""
        有监督学习：
            在有监督学习中，训练数据包含了输入特征以及相应的输出标签。模型通过学习这些已知的输入-输出对来预测未知数据的输出。
            这类方法的目标是找到一个映射函数，使得给定新的输入数据时能够准确地预测输出。常见的有监督学习任务包括分类（如垃圾邮件检测）和回归（如房价预测）
    """
def train_supervised(predict_text):
    model = fasttext.train_supervised('cooking.stackexchange.txt', epoch=10)
    print('词汇长度: ', len(model.words))

    print(model.predict(predict_text))


"""
    有监督学习和无监督学习最主要的区别在于输入数据是否带标签
"""
if __name__ == '__main__':
    # train_unsupervised()
    train_supervised("Cake sinks in the middle when baking. Only happens when I make a Coca-Cola Cake")

