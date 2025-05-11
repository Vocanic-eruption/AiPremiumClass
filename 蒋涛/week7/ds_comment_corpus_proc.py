import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle

def build_vocab_from_documents(ds_comments):
    """
    构建词汇表，将输入的文本数据中的所有唯一单词映射到一个唯一的整数索引。

    :param ds_comments: 包含多个文本评论的列表，每个评论是一个字符串。
    :return: 一个字典，键为单词，值为对应的整数索引。
    """
    # 初始化一个空集合，用于存储所有唯一的单词
    vocab = set()
    # 遍历输入的评论列表
    for c in ds_comments:
        # 将当前评论按空格分割成单词列表，并将这些单词添加到集合中
        # 集合的特性会自动去重
        vocab.update(c.split())
    
    # 创建一个包含特殊标记 '<PAD>' 和 '<UNK>' 的列表
    # '<PAD>' 通常用于填充序列，使不同长度的序列长度一致
    # '<UNK>' 用于表示未知单词
    # 然后将集合中的唯一单词转换为列表并添加到后面
    tokens = ['<PAD>', '<UNK>'] + list(vocab)
    # 使用字典推导式创建一个字典，将每个单词映射到一个唯一的整数索引
    new_tokens = {t: i for i, t in enumerate(tokens)}
    return new_tokens


def build_collate_func(comments_vocab):
    """
    构建数据加载器的collate_fn函数，该函数用于处理从数据集中取出的一个批次的数据，
    将文本数据转换为模型可接受的张量格式，并对序列进行填充。

    :param comments_vocab: 词汇表字典，键为单词，值为对应的整数索引。
    :return: 一个可调用的collate_fn函数，用于数据加载器处理批次数据。
    """
    def collate_fn(batch):
        # 初始化两个空列表，分别用于存储评论的张量表示和对应的标签
        comments, labels = [], []
        # 遍历批次中的每个样本
        # 这里原代码使用 enumerate 有误，会导致取到的 item 是 (索引, 样本) 格式，应直接遍历 batch
        for item in batch:
            # 将当前样本中的文本分词，并将每个词转换为对应的整数索引
            # 使用 comments_vocab.get 方法获取单词的索引，如果单词不在词汇表中，则使用 '<UNK>' 的索引
            # 同时过滤掉空字符串
            token_index = torch.tensor([comments_vocab.get(t, comments_vocab['<UNK>']) for t in item[0].split() if t != ' '])
            # 将当前样本的张量表示添加到 comments 列表中
            comments.append(token_index)
            # 假设 item[1] 是当前样本的标签，将其添加到 labels 列表中
            labels.append(item[1])
        # 使用 pad_sequence 函数对评论序列进行填充，使它们具有相同的长度
        # batch_first=True 表示输出的张量中批次维度在第一维
        # padding_value 指定填充的值为 '<PAD>' 的索引
        comments = pad_sequence(comments, batch_first=True, padding_value=comments_vocab['<PAD>'])
        # 将标签列表转换为 torch.Tensor 类型，数据类型为 torch.int64
        labels = torch.tensor(labels,  dtype=torch.int64)
        return comments, labels
    
    return collate_fn

def get_dataloader():
    """
    该函数的主要功能是从 pickle 文件中加载分词后的评论数据，构建词汇表，创建数据集和数据加载器，
    并打印数据加载器返回的第一批数据。

    :return: 无返回值，主要作用是打印数据加载器的第一批数据。
    """
    # 以二进制读取模式打开 pickle 文件，加载分词后的评论数据
    with open('week7_rnn文本分类/week7/ds_comments_jieba.bin','rb') as f:
        comments_jieba, labels = pickle.load(f)
    # 打印从 pickle 文件中加载的对象数量
    print(f"pickle 文件返回了 {len(comments_jieba)} 个对象")
    # 调用 build_vocab_from_documents 函数构建词汇表
    # 该词汇表将每个单词映射到一个唯一的整数索引
    vocab = build_vocab_from_documents(comments_jieba)
    
    # 自定义数据集，将分词后的评论数据转换为适合 DataLoader 处理的格式
    # 这里使用 zip 函数将评论数据打包成元组列表，但可能需要添加对应的标签数据
    ds = list(zip(comments_jieba, labels))

    # 创建数据加载器，用于批量加载数据
    # batch_size=10 表示每个批次加载 10 个样本
    # collate_fn=build_collate_func(vocab) 表示使用自定义的 collate 函数处理批次数据
    # shuffle=True 表示在每个训练周期开始时打乱数据顺序
    dl = DataLoader(ds, batch_size=10, collate_fn=build_collate_func(vocab), shuffle=True)

    # 遍历数据加载器，获取第一批数据
    for item in dl:
        # 打印第一批数据的第一个和第二个元素，通常为输入数据和标签
        print(item[0], item[1])
        # 只打印第一批数据，然后跳出循环
        break


if __name__ == '__main__':
    get_dataloader()
    