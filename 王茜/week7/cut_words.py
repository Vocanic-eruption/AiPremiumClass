import jieba
import sentencepiece as spm


def cut_words_jieba(texts):
    texts_split = []
    for text in texts:
        text_split = jieba.lcut(text[0])
        texts_split.append((text_split, text[1]))
    return texts_split


def cut_words_sentence_pieces_train(file, vocab_size):
    spm.SentencePieceTrainer.Train(
        input=file,
        vocab_size=vocab_size,
        model_prefix='spm_words_cut'
    )

def cut_words_sentence_pieces_test(texts):
    s = spm.SentencePieceProcessor(model_file='spm_words_cut.model')
    texts_split = []
    for text in texts:
        text_split = s.encode(text[0])
        texts_split.append((s.id_to_piece(text_split), text[1]))
    return texts_split

if __name__ == '__main__':
    import pandas as pd
    file = 'douban_comments.csv'
    vocab_size = 200000
    cut_words_sentence_pieces_train(file, vocab_size)


