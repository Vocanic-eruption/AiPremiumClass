import pickle

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

votes_map = {
    '0': 0,
    '1': 1
}

def c2idx(comment, w2idx):
    return [w2idx.get(word, w2idx['UNK']) for word in comment]


def convert_data(batch_data):
    comments, votes = [], []
    for comment, vote in batch_data:
        comments.append(torch.tensor(c2idx(comment, w2idx)))
        votes.append(votes_map[vote])
    # 将评论和标签转换为tensor
    comments = pad_sequence(comments, batch_first=True, padding_value=0)
    labels = torch.tensor(votes)
    return comments, labels


class Comments_Classifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) # 指定填充值，训练时忽略
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        assert input_ids.max() < self.embedding.num_embeddings, \
            f"输入索引{input_ids.max()}超出嵌入层范围{self.embedding.num_embeddings}"
        embedded = self.embedding(input_ids)
        output, (hidden, _) = self.rnn(embedded)
        output = self.fc(output[:, -1, :])
        return output


if __name__ == '__main__':
    cut_type = 'spm'
    comments = pickle.load(open(f'{cut_type}_db_comments.pkl', 'rb'))
    w2idx = pickle.load(open(f'{cut_type}_db_w2idx.pkl', 'rb'))

    train_flag = 1   # 模型训练
    test_flag = 0  # 模型预测
    device = 'cpu'
    embedding_dim = 100
    hidden_size = 128
    num_classes = 2
    num_epochs = 10
    batch_size = 500

    # 构建DataLoader
    dataloader = DataLoader(comments,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=convert_data,
                            drop_last=True)

    # 构建模型
    model = Comments_Classifier(len(w2idx), embedding_dim, hidden_size, num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    print(model)
    # 训练模型
    for epoch in range(num_epochs):
        i = 0
        for batch, labels in dataloader:
            batch = batch.to(device)
            labels = labels.to(device)
            pred = model(batch)
            loss = criterion(pred, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 20 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step[{i}],  Loss: {loss.item():.4f}')
            i = i + 1

    # 保存模型
    torch.save(model.state_dict(), f'{cut_type}_db_comments_classifier.pth')