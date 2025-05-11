import torch
from torch.utils.data import DataLoader
import json
import pickle
from torch.nn.utils.rnn import pad_sequence


def read_data():
    en_doc_arr, de_doc_arr = [], []

    with open('./cmn.txt', 'r', encoding='utf-8') as f:
        lines = f.read().split("\n")
        for line in lines:
            # print(repr(row))
            if line == '':
                continue
            en_doc, de_doc = line.split("\t")

            en_doc = en_doc.replace(",", "").replace(".", "").replace("!", "").replace("?", "")
            de_doc = de_doc.replace("，", "").replace("。", "").replace("！", "").replace("？", "")

            en_doc = en_doc.split()
            # de_doc = ['BOS'] + list(jieba.cut_for_search(de_doc)) + ['EOS']
            de_doc = ['BOS'] + list(de_doc) + ['EOS']

            en_doc_arr.append(en_doc)
            de_doc_arr.append(de_doc)
    return en_doc_arr, de_doc_arr


class Vocabulary:
    def __init__(self, vocab):
        self.vocab = vocab

    @classmethod
    def build_vocab(cls, docs):
        tokens_set = set()
        for doc in docs:
            tokens_set.update(doc)

        vocab = {word: idx for idx, word in enumerate(tokens_set)}
        return cls(vocab)


class BatchDataProcessor:
    def __init__(self, enc_vocab, dec_vocab):
        self.enc_vocab = enc_vocab
        self.dec_vocab = dec_vocab

    def get_word_idx(self, batch_data):
        encode_inputs = []
        decode_inputs = []
        targets = []
        for data in batch_data:
            en_doc, de_doc = data
            enc_idx = [self.enc_vocab[word] for word in en_doc]
            dec_idx = [self.dec_vocab[word] for word in de_doc]

            encode_inputs.append(torch.tensor(enc_idx))
            decode_inputs.append(torch.tensor(dec_idx[:-1]))
            targets.append(torch.tensor(dec_idx[1:]))

        encode_inputs = pad_sequence(encode_inputs, batch_first=True)
        decode_inputs = pad_sequence(decode_inputs, batch_first=True)
        targets = pad_sequence(targets, batch_first=True)
        return encode_inputs, decode_inputs, targets


if __name__ == '__main__':
    # # 读数据
    # enc_docs, dec_docs = read_data()
    #
    # # 构建词汇表
    # enc_vocab = Vocabulary.build_vocab(enc_docs)
    # dec_vocab = Vocabulary.build_vocab(dec_docs)
    #
    # # 编码+解码(构成训练样本)
    # dataset = list(zip(enc_docs, dec_docs))
    #
    # batch_processor = BatchDataProcessor(enc_vocab.vocab, dec_vocab.vocab)
    # # DataLoader
    # dl = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=batch_processor.get_word_idx)
    # # for batch in dl:
    # #     print(batch)
    # #     break
    #
    # # 数据集缓存
    # with open('encode.json', 'w', encoding='utf-8') as f:
    #     json.dump(enc_docs, f)
    #
    # with open('decoder.json', 'w', encoding='utf-8') as f:
    #     json.dump(dec_docs, f)
    #
    # # 词典表保存
    # with open('vocab.bin', 'wb') as f:
    #     pickle.dump((enc_vocab, dec_vocab), f)

    with open('vocab.bin', 'rb') as f:
        enc_vocab, dec_vocab = pickle.load(f)
    print(enc_vocab)
    # print(dec_vocab)
