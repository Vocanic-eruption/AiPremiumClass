import pickle
from week07_classify import Config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vocabulary:
    def __init__(self):
        self.vocabulary = set()
        self.word2index = {}

    def load_build(self, path):
        with open(path, 'rb') as f:
            comments_data = pickle.load(f)
        print("示例数据:", comments_data[0])

        for comment in comments_data:
            self.vocabulary.update(comment[0])
        print("词汇表大小:", len(self.vocabulary))

        self.word2index = {word: idx + 2 for idx, word in enumerate(self.vocabulary)}
        self.word2index['<PAD>'] = 0
        self.word2index['<UNK>'] = 1


class MyEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(MyEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

    def forward(self, input_ids):
        return self.embedding(input_ids)


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = MyEmbedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, text):
        embedded = self.embedding(text)
        _, hidden = self.rnn(embedded)
        output = self.fc(hidden.squeeze(0))
        return output


def text_pipeline(text, word2index):
    return [word2index.get(word, word2index['<UNK>']) for word in text]


def label_pipeline(label):
    return int(label)


def collate_batch(batch):
    label_list, text_list = [], []
    for (commet, label) in batch:
        label_list.append(label_pipeline(label))
        processed_text = torch.tensor(text_pipeline(commet, vocab.word2index), dtype=torch.int64)
        text_list.append(processed_text)

    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = pad_sequence(text_list, batch_first=True, padding_value=vocab.word2index['<PAD>'])
    return text_list, label_list


if __name__ == '__main__':
    # 1. 构建词汇表
    vocab = Vocabulary()
    vocab.load_build(Config.SP_PATH)

    # 2. 准备数据
    with open(Config.JIEBA_DATA_PATH, 'rb') as f:
        comments_data = pickle.load(f)

    dataloader = DataLoader(
        comments_data,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch
    )

    # 5. 初始化模型
    model = TextClassifier(
        vocab_size=len(vocab.word2index),
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_size=Config.HIDDEN_SIZE,
        num_classes= 2
    ).to(device)

    # 6. 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 7. 训练循环
    print("开始训练...")
    for epoch in range(Config.NUM_EPOCHS):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        model.train()
        for texts, labels in dataloader:
            texts = texts.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print(loss.item())



