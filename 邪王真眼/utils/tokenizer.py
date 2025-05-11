from enum import Enum
from typing import List, Set


class SegmenterType(Enum):
    JIEBA = 'jieba'
    SENTENCE_PIECE = 'sentence_Piece'
    MAXIMUM_FORWORD = 'maximum_forward'


class Tokenizer:
    def __init__(self, texts: List[str], seg_type: SegmenterType = SegmenterType.JIEBA):
        self.seg_type = seg_type
        self.vocab_to_index = {"UNK": 0, "PAD": 1}
        self.index_to_vocab = {0: "UNK", 1: "PAD"}
        self.next_idx = 2
        
        tokenized_texts = self._tokenize_texts(texts)
        self._build_vocab(tokenized_texts)
        self.max_len = max(len(tokens) for tokens in tokenized_texts) if tokenized_texts else 0
    
    def _tokenize_texts(self, texts: List[str]) -> List[List[str]]:
        tokenized_texts = []
        
        if self.seg_type == SegmenterType.JIEBA:
            from .jieba_segmenter import JiebaSegmenter
            for text in texts:
                tokens = list(JiebaSegmenter.cut(text))
                tokenized_texts.append(tokens)
                
        elif self.seg_type == SegmenterType.SENTENCE_PIECE:
            from .sentence_piece_segmenter import SentencePieceSegmenter
            SentencePieceSegmenter.train(texts)
            for text in texts:
                tokens = SentencePieceSegmenter.cut(text)
                tokenized_texts.append(tokens)
        
        elif self.seg_type == SegmenterType.MAXIMUM_FORWORD:
            from .maximum_forward_segmenter import MaximumForwardSegmenter
            for text in texts:
                tokens = MaximumForwardSegmenter.max_forward_cut2(text)
                tokenized_texts.append(tokens)

        return tokenized_texts
    
    def _build_vocab(self, tokenized_texts: List[List[str]]):
        vocab: Set[str] = set()
        for tokens in tokenized_texts:
            vocab.update(tokens)
        
        for word in vocab:
            if word not in self.vocab_to_index:
                self.vocab_to_index[word] = self.next_idx
                self.index_to_vocab[self.next_idx] = word
                self.next_idx += 1

    def encode(self, text: str, max_len: int = None) -> List[int]:
        if max_len is None:
            max_len = self.max_len
        tokens = self._tokenize_text(text)
        indices = [self.vocab_to_index.get(t, 0) for t in tokens]  # 0=UNK
        pad_len = max_len - len(indices)
        return indices + [1] * pad_len if pad_len > 0 else indices[:max_len]  # 1=PAD
    
    def _tokenize_text(self, text: str) -> List[str]:
        if self.seg_type == SegmenterType.JIEBA:
            from .jieba_segmenter import JiebaSegmenter
            return JiebaSegmenter.cut(text)
        elif self.seg_type == SegmenterType.SENTENCE_PIECE:
            from .sentence_piece_segmenter import SentencePieceSegmenter
            return SentencePieceSegmenter.cut(text)
        elif self.seg_type == SegmenterType.MAXIMUM_FORWORD:
            from .maximum_forward_segmenter import MaximumForwardSegmenter
            return MaximumForwardSegmenter.max_forward_cut2(text)
