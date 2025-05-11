import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence  # 长度不同张量填充为相同长度
import jieba

# 构建包含所有词的词嵌入矩阵
def build_from_doc(doc):
    vocab = set()
    for line in doc:
        vocab.update(line[0])

    vocab =  ['PAD','UNK'] + list(vocab)  # PAD: padding, UNK: unknown
    w2idx = {word: idx for idx, word in enumerate(vocab)}
    return w2idx

# 自定义数据转换方法(callback function)回调函数
# 该函数会在每个batch数据加载时被调用
def convert_data(batch_data):
    comments, votes = [],[]
    # 分别提取评论和标签
    for comment, vote in batch_data:
        comments.append(torch.tensor([vocab_dict.get(word, vocab_dict['UNK'])
                                        for word in comment]))
        votes.append(vote)

# 将评论和标签转换为tensor
    commt = pad_sequence(comments, batch_first=True,
                            padding_value=vocab_dict['PAD'])  # 填充为相同长度
    labels = torch.tensor(votes)
    # 返回评论和标签
    return commt, labels

class Comments_Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # padding_idx=0
        self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        # embedded: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)
        # output: (batch_size, seq_len, hidden_size)
        output, hidden = self.rnn(embedded)
        output = self.fc(output[:, -1, :])  # 取最后一个时间步的输出
        return output

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载训练语料
    with open('moves_comments.pkl','rb') as f:
        moves_comments = pickle.load(f)

    # print(moves_comments[0]) # 元组，(评论，标签)

    # 构建词汇表字典 {word: idx}
    vocab_dict = build_from_doc(moves_comments)
    # print('词汇表字典大小:', len(vocab_dict)) # 162935
    # print (vocab_dict)

    moves_comments_train = moves_comments[:int(len(moves_comments)*0.8)]
    moves_comments_eval = moves_comments[int(len(moves_comments)*0.8):]

    # 通过Dataset构建DataLoader
    train_data = DataLoader(moves_comments_train, batch_size=512, shuffle=True,
                            collate_fn=convert_data)
    eval_data = DataLoader(moves_comments_eval, batch_size=128, shuffle=True,
                            collate_fn=convert_data)

    vocab_size = len(vocab_dict)
    embedding_dim = 100
    hidden_size = 128
    num_classes = 2

    model = Comments_Classifier(vocab_size, embedding_dim, hidden_size, num_classes)
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 5
    for epoch in range(num_epochs):
        for i, (cmt, lbl) in enumerate(train_data):
            cmt = cmt.to(device)
            lbl = lbl.to(device)

            # 前向传播
            outputs = model(cmt)
            loss = criterion(outputs, lbl)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}],Step [{i+1}/{len(train_data)}],Loss: {loss.item():.4f}')

    # 评估模型
    with torch.no_grad():
        correct = 0
        total = 0
        for cmt, lbl in eval_data:
            cmt = cmt.to(device)
            lbl = lbl.to(device)

            outputs = model(cmt)
            _, predicted = torch.max(outputs.data, 1)

            total += lbl.size(0)
            correct += (predicted == lbl).sum().item()

        print(f'Evaluation Accuracy: {correct/total:.4f}')

    # 测试模型
    comment1 = '''鹰眼作为本片的第一男主”有理有据。有感情戏，
    有主公技，有不止一个妹，有感化傻逼的男主式语录，有狂立flag而不倒的金手指
    ，有死基友不死本大爷的神功护体。。他合同是不是结束了？？？复联2——\鹰眼巨巨凯瑞me/    还我快银！！！'''
    comment2 = ' 不出意料得烂，喜欢这部电影的孩子，大概也喜欢变4……看到一半就不想看了，太无聊了，演员演技也很差'

    # 将评论转换为索引
    comment1_idx = torch.tensor([vocab_dict.get(word, vocab_dict['UNK']) for word in jieba.lcut(comment1)])
    comment2_idx = torch.tensor([vocab_dict.get(word, vocab_dict['UNK']) for word in jieba.lcut(comment2)])
    # 将评论转换为tensor
    comment1_idx = comment1_idx.unsqueeze(0).to(device)  # 添加batch维度
    comment2_idx = comment2_idx.unsqueeze(0).to(device)  # 添加batch维度

    # 加载模型
    model = Comments_Classifier(len(vocab_dict), embedding_dim, hidden_size, num_classes)
    model.load_state_dict(torch.load('comments_classifier.pth'))
    model.to(device)

    # 模型推理
    pred1 = model(comment1_idx)
    pred2 = model(comment2_idx)

    # 取最大值的索引作为预测结果
    pred1 = torch.argmax(pred1, dim=1).item()
    pred2 = torch.argmax(pred2, dim=1).item()
    print(f'评论1预测结果: {pred1}')
    print(f'评论2预测结果: {pred2}')

    # 使用 spm 分词
    import sentencepiece as spm
    import csv
    file_name = 'douban_moves_comments.csv'
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            comment = row['comment']
            with open('comments.txt', 'a', encoding='utf-8') as f:
                f.write(comment + '\n')

    spm.SentencePieceTrainer.Train(input='comments.txt', model_prefix='comments',
                                   vocab_size=700000, character_coverage=1.0, model_type='unigram')

    sp = spm.SentencePieceProcessor(model_file='comments.model')

    moves_comments1 = []
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            star = int(row['Star'])
            comment = row['Comment']
            # print(star)
            # print(comment)
            if star in [1, 2, 4, 5]:
                # print(comment)
                words = sp.EncodeAsPieces(comment)
                moves_comments1.append((words, 1 if star > 3 else 0 ))

    moves_comments1 = [c for c in moves_comments1 if len(c[0]) in range(10, 80)]

    # 构建词汇表字典 {word: idx}
    vocab_dict1 = build_from_doc(moves_comments1)
    # print('词汇表字典大小:', len(vocab_dict)) # 162935
    # print (vocab_dict)

    moves_comments_train1 = moves_comments[:int(len(moves_comments1)*0.8)]
    moves_comments_eval1 = moves_comments[int(len(moves_comments1)*0.8):]

    # 通过Dataset构建DataLoader
    train_data1 = DataLoader(moves_comments_train1, batch_size=512, shuffle=True,
                            collate_fn=convert_data)
    eval_data1 = DataLoader(moves_comments_eval1, batch_size=128, shuffle=True,
                            collate_fn=convert_data)

    vocab_size1 = len(vocab_dict1)
    embedding_dim = 100
    hidden_size = 128
    num_classes = 2

    model1 = Comments_Classifier(vocab_size1, embedding_dim, hidden_size, num_classes)
    model1.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model1.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 5
    for epoch in range(num_epochs):
        for i, (cmt, lbl) in enumerate(train_data1):
            cmt = cmt.to(device)
            lbl = lbl.to(device)

            # 前向传播
            outputs = model1(cmt)
            loss = criterion(outputs, lbl)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}],Step [{i+1}/{len(train_data1)}],Loss: {loss.item():.4f}')
