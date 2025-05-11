import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import jieba
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


# 配置参数
class Config:
    data_path = "/kaggle/input/doubanmovieshortcomments/DMSC.csv"
    max_seq_len = 200  # 最大序列长度
    batch_size = 64  # 批次大小
    embedding_dim = 256  # 词向量维度
    hidden_dim = 128  # 隐藏层维度
    num_classes = 2  # 分类类别
    lr = 0.001  # 学习率
    epochs = 10  # 训练轮次
    min_freq = 3  # 最小词频
    train_ratio = 0.8  # 训练集比例
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# 数据预处理类
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.word2idx = {}
        self.idx2word = {}

    @staticmethod
    def clean_text(text):
        """文本清洗函数"""
        # 去除特殊字符、保留中文和数字
        text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)
        # 合并多个空格
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def build_vocab(self, texts):
        """构建词典"""
        word_counter = Counter()
        for text in texts:
            words = jieba.lcut(text)
            word_counter.update(words)

        # 过滤低频词
        vocab = [word for word, count in word_counter.items() if count >= self.config.min_freq]
        # 添加特殊标记
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        for idx, word in enumerate(vocab, start=2):
            self.word2idx[word] = idx
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def text_to_sequence(self, text):
        """文本转数字序列"""
        words = jieba.lcut(text)
        sequence = [self.word2idx.get(word, 1) for word in words]
        # 截断或填充
        if len(sequence) > self.config.max_seq_len:
            sequence = sequence[:self.config.max_seq_len]
        else:
            sequence += [0] * (self.config.max_seq_len - len(sequence))
        return sequence


# 自定义数据集类
class CommentDataset(Dataset):
    def __init__(self, texts, labels, processor):
        self.texts = texts
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        sequence = self.processor.text_to_sequence(self.texts[idx])
        return (
            torch.tensor(sequence, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


# 文本分类模型
class TextClassifier(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(config.embedding_dim,
                            config.hidden_dim,
                            batch_first=True,
                            bidirectional=True,
                            num_layers=2)
        self.fc = nn.Linear(config.hidden_dim * 2, config.num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        # 拼接双向LSTM输出
        hidden = self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1))
        return self.fc(hidden)


# 训练流程
def main():
    # 初始化配置
    config = Config()
    processor = DataProcessor(config)

    # 数据加载与预处理
    df = pd.read_csv(config.data_path)
    print("原始数据示例：\n", df.head())

    # 筛选评分并生成标签
    df = df[df["Star"].isin([1, 2, 4, 5])]  # 保留有效评分
    df["label"] = df["Star"].apply(lambda x: 1 if x <= 2 else 0)
    print("\n标签分布：\n", df["label"].value_counts())

    # 清洗文本
    df["cleaned_text"] = df["Comment"].apply(processor.clean_text)

    # 划分数据集
    train_df, test_df = train_test_split(
        df,
        test_size=1 - config.train_ratio,
        random_state=42,
        stratify=df["label"]
    )

    # 构建词典
    processor.build_vocab(train_df["cleaned_text"])
    print(f"\n构建词典完成，词汇量：{len(processor.word2idx)}")

    # 创建DataLoader
    train_dataset = CommentDataset(
        train_df["cleaned_text"].tolist(),
        train_df["label"].tolist(),
        processor
    )
    test_dataset = CommentDataset(
        test_df["cleaned_text"].tolist(),
        test_df["label"].tolist(),
        processor
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=config.batch_size)

    # 初始化模型
    model = TextClassifier(config, len(processor.word2idx)).to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # 训练循环
    best_acc = 0.0
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for inputs, labels in tqdm(train_loader,
                                   desc=f"Epoch {epoch + 1}/{config.epochs}"):
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 评估验证集
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(config.device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pth")

    print(f"\n最佳测试准确率：{best_acc:.4f}")


if __name__ == "__main__":
    main()
