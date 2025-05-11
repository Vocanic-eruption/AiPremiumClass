import pickle
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sentencepiece as spm


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, nums_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_dim)
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, nums_classes)

    def forward(self, input_idxs):
        word_matrix = self.embedding(input_idxs)
        out, hidden = self.rnn(word_matrix)
        return self.fc(out[:, -1, :])


def create_vocab_from_docs(docs):
    vocab = set()
    votes = []
    for doc, vote in docs:
        vocab.update(doc)
        votes.append(vote)
    vocab_list = ['<unk>', 'PAD'] + list(vocab)
    word2idx = {word: idx for idx, word in enumerate(vocab_list)}
    return word2idx


def padding_word_matrix(batch_data, word2idx):
    comments, targets = [], []
    for doc, vote in batch_data:
        comments.append(torch.tensor([word2idx.get(word, word2idx['<unk>']) for word in doc]))
        targets.append(vote)

    common_len_comments = pad_sequence(comments, batch_first=True, padding_value=word2idx['PAD'])  # 填充为相同长度
    return common_len_comments, torch.tensor(targets, dtype=torch.long)


def create_vocab_from_docs_spm():
    sp = spm.SentencePieceProcessor(model_file='comments_spm.model')
    vocab = {sp.IdToPiece(i): i for i in range(sp.vocab_size())}
    vocab['PAD'] = sp.vocab_size()
    return vocab


if __name__ == '__main__':
    hidden_size = 128
    embedding_dim = 100
    nums_classes = 3
    epochs = 40

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter("spm")

    # 加载语料库
    with open('fixed_comments.pkl', 'rb') as f:
        fixed_comments = pickle.load(f)

    # 构建词汇表
    # word2idx = create_vocab_from_docs(fixed_comments)
    word2idx = create_vocab_from_docs_spm()

    # 定义模型结构(词向量嵌入)
    train_model = TextClassifier(len(word2idx), hidden_size, embedding_dim, nums_classes)
    train_model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(train_model.parameters(), lr=0.001)

    # 数据转化
    train_data = DataLoader(fixed_comments,
                            batch_size=32,
                            shuffle=True,
                            collate_fn=lambda batch_data: padding_word_matrix(batch_data, word2idx))

    # 训练
    total = 0
    for epoch in range(epochs):
        for batch_data, batch_target in train_data:
            train_model.train()
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            output = train_model(batch_data)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
            total += 1

            writer.add_scalar('text classifier loss', loss, global_step=total)
        print(f'epoch:{epoch + 1}, loss:{loss.item():.4f}')

    writer.close()

    # 保存模型
    torch.save(train_model.state_dict(), 'text_classifier.pth')
    torch.save(word2idx, 'vocab_idx.pth')
