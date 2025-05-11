import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter


def read_data(enc_file, dec_file):
    enc_data, dec_data = [], []
    with open(enc_file, 'r', encoding='utf-8') as f_enc:
        enc_lines = f_enc.read().splitlines()
    with open(dec_file, 'r', encoding='utf-8') as f_dec:
        dec_lines = f_dec.read().splitlines()
    assert len(enc_lines) == len(dec_lines), f"数据未对齐: {enc_file} vs {dec_file}"  # 检查数据长度是否一致

    for enc_line, dec_line in zip(enc_lines, dec_lines):
        enc_line = enc_line.strip()
        dec_line = dec_line.strip()
        if not enc_line or not dec_line:
            continue

        enc_tokens = list(enc_line.replace(' ', ''))
        dec_tokens = ['BOS'] + list(dec_line.replace(' ', '')) + ['EOS']
        enc_data.append(enc_tokens)
        dec_data.append(dec_tokens)
    return enc_data, dec_data


def batch_proc_factory(enc_vocab, dec_vocab):
    def collate_fn(batch):
        enc_batch, dec_batch, label_batch = [], [], []

        for enc_tokens, dec_tokens in batch:
            enc_ids = [enc_vocab[tk] for tk in enc_tokens]
            dec_ids = [dec_vocab[tk] for tk in dec_tokens]

            enc_batch.append(torch.tensor(enc_ids))
            dec_batch.append(torch.tensor(dec_ids[:-1]))
            label_batch.append(torch.tensor(dec_ids[1:]))
        enc_padded = pad_sequence(enc_batch, batch_first=True, padding_value=0)
        dec_padded = pad_sequence(dec_batch, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(label_batch, batch_first=True, padding_value=0)

        return enc_padded, dec_padded, labels_padded

    return collate_fn


class Vocabulary:
    SPECIAL_TOKENS = {'PAD': 0, 'UNK': 1, 'BOS': 2, 'EOS': 3}

    def __init__(self, token_dict):
        self.token2idx = token_dict
        self.idx2token = {v: k for k, v in token_dict.items()}

    @classmethod
    def from_documents(cls, documents):
        all_tokens = set()
        for doc in documents:
            all_tokens.update(doc)

        # 构建词汇字典
        token_dict = cls.SPECIAL_TOKENS.copy()
        current_idx = len(token_dict)
        for token in all_tokens:
            if token not in token_dict:
                token_dict[token] = current_idx
                current_idx += 1

        return cls(token_dict)


if __name__ == '__main__':
    train_enc, train_dec = read_data('./train/in.txt', './train/out.txt')
    enc_vocab = Vocabulary.from_documents(train_enc)
    dec_vocab = Vocabulary.from_documents(train_dec)
    # print(编码样本数量: {len(train_enc)}")
    # print(f"解码器样本数量: {len(train_dec)}")

    train_loader = DataLoader(
        list(zip(train_enc, train_dec)),
        batch_size=32,
        shuffle=True,
        collate_fn=batch_proc_factory(enc_vocab, dec_vocab)
    )

    # sample_enc, sample_dec, sample_labels = next(iter(train_loader))
    # print(f"编码器输入: {sample_enc.shape}")

    # # 保存词汇表
    # torch.save({
    #     'encoder': enc_vocab.token2idx,
    #     'decoder': dec_vocab.token2idx
    # }, 'full_vocabs.pt')

    import json
    import pickle
    with open('encoder.json', 'w', encoding='utf-8') as f:
        json.dump(train_enc, f, ensure_ascii=False)
    with open('decoder.json', 'w', encoding='utf-8') as f:
        json.dump(train_dec, f, ensure_ascii=False)

    with open('vocab.bin', 'wb') as f:
        pickle.dump((enc_vocab.token2idx, dec_vocab.token2idx), f)
