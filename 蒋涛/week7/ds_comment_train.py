import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim 
from tqdm import tqdm
import random
import pickle
from torch.utils.tensorboard import SummaryWriter

from ds_comment_classifier_model import CommentsClassifier
from ds_comment_corpus_proc import build_vocab_from_documents, build_collate_func

try:
    mps_available = torch.mps.is_available()
except AttributeError:
    mps_available = False

# 1. SummaryWriter全局对象
writer = SummaryWriter()
# 跟踪相关变量 
train_loss_cnt = 0
val_loss_cnt = 0
val_acc_cnt = 0

# hyperparameter 超参数
BATCH_SIZE=128       # 批次大小
EPOCHS=5           # 训练轮数
EMBEDDING_SIZE=200  # 词向量维度
RNN_HIDDEN_SIZE=200 # RNN隐藏层维度
LEARN_RATE=1e-3     # 学习率 
NUM_LABELS=2        # label数量

# 设备选择
# device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.mps.is_available() else 'cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if mps_available else 'cpu'))

def train_test_split(X, y, test_size=0.2):
    """
    划分数据集为训练集和测试集。
    :param X: 输入特征，通常是一个列表或数组。
    :param y: 标签，通常是一个列表或数组。
    :param test_size: 测试集占比，默认为0.2。
    :return: 训练集和测试集的特征和标签。
    """
    # 计算测试集的大小
    test_size = int(len(X) * (1 - test_size))
    
    test_index = list(range(len(X)))
    random.shuffle(test_index)
    x_train = [X[i] for i in test_index[:test_size]]
    y_train = [y[i] for i in test_index[:test_size]]
    x_test = [X[i] for i in test_index[test_size:]]
    y_test = [y[i] for i in test_index[test_size:]]
    return x_train, y_train, x_test, y_test

def train_step(model, tokens, labels, criterion):
    logits = model(tokens)
    loss = criterion(logits, labels)
    return loss

def train_model(model, train_dl, criterion, optimizer, epoch):
    global train_loss_cnt
    model.train()
    tpbar = tqdm(train_dl)
    for tokens, labels in tpbar:
        optimizer.zero_grad()
        tokens, labels = tokens.to(device), labels.to(device)
        loss = train_step(model, tokens, labels, criterion)
        loss.backward()
        optimizer.step()
        tpbar.set_description(f'epoch:{epoch+1} train_loss:{loss.item():.4f}')
         # tensorboard跟踪记录
        writer.add_scalar('train_loss', loss.item(), train_loss_cnt)
        train_loss_cnt += 1


def validate(model, val_dl, criterion, epoch):
    global val_loss_cnt, val_acc_cnt
    model.eval()
    tpbar = tqdm(val_dl)
    total_loss = 0
    total_acc = 0
    for tokens, labels in tpbar:
        tokens,labels = tokens.to(device), labels.to(device)
        loss, logits = validate_step(model, tokens, labels, criterion)
        tpbar.set_description(f'epoch:{epoch+1} val_loss:{loss.item():.4f}')
        
        total_loss += loss.item()
        total_acc += (logits.argmax(dim=1) == labels).float().mean()
    
    # tensorboard跟踪记录
    writer.add_scalar('val_avg_loss', total_loss / len(val_dl), val_loss_cnt)
    val_loss_cnt += 1
    # 计算准确率
    writer.add_scalar('val_acc', total_acc / len(val_dl), val_acc_cnt)
    val_acc_cnt += 1

def validate_step(model, tokens, labels, criterion):
    with torch.no_grad():
        logits = model(tokens)
        loss = criterion(logits, labels)
    return loss, logits

def run_train():
    # 数据准备
    with open('week7_rnn文本分类/week7/ds_comments_jieba.bin','rb') as f:
        comments_jieba, labels = pickle.load(f)

    vocab = build_vocab_from_documents(comments_jieba)

    # 数据拆分
    x_train, y_train, x_test, y_test = train_test_split(comments_jieba, labels)
    x_valid, y_valid, x_test, y_test = train_test_split(x_test, y_test, test_size=0.5)
    
    # 自定义Dataset处理文本数据转换
    train_ds = list(zip(x_train, y_train))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, 
                          collate_fn=build_collate_func(vocab), shuffle=True)
    valid_ds = list(zip(x_valid, y_valid))
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, 
                          collate_fn=build_collate_func(vocab), shuffle=True)
    test_ds = list(zip(x_test, y_test))
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, 
                         collate_fn=build_collate_func(vocab), shuffle=False)

    # 模型构建
    model = CommentsClassifier(
        vocab_size=len(vocab), 
        embedding_dim=EMBEDDING_SIZE, 
        hidden_size=RNN_HIDDEN_SIZE, 
        num_labels=NUM_LABELS
    )
    model.to(device)

    # loss function、optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
    criterion = nn.CrossEntropyLoss()

    # 训练
    for epoch in range(EPOCHS):
        train_model(model, train_dl, criterion, optimizer, epoch)
        validate(model, valid_dl, criterion, epoch)

    # 保存模型参数
    torch.save(
        {'model_state': model.state_dict,
         'model_vocab': vocab}, 'ds_model_objs2.bin')
    
if __name__ == '__main__':
    run_train()