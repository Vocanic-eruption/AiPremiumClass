import os
import jieba
import shutil
import tempfile
import sentencepiece as spm

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional


class Tokenizer(ABC):
    """Abstract base class for tokenizers"""
    def __init__(self):
        self.vocab_to_index: Dict[str, int] = {}
        self.index_to_vocab: Dict[int, str] = {}
        self.max_len: int = 0

    @abstractmethod
    def train(self, texts: List[str]) -> None:
        """Train the tokenizer on given texts"""
        pass

    @abstractmethod
    def encode(self, text: str, max_len: Optional[int] = None) -> List[int]:
        """Convert text to sequence of indices"""
        pass

    def get_vocab(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Get vocabulary mappings"""
        return self.vocab_to_index, self.index_to_vocab

class JiebaTokenizer(Tokenizer):
    """Jieba-based word tokenizer"""
    def __init__(self):
        super().__init__()
        # Initialize with special tokens
        self.vocab_to_index = {"UNK": 0, "PAD": 1}
        self.index_to_vocab = {0: "UNK", 1: "PAD"}
        self.next_idx = 2

    def train(self, texts: List[str]) -> None:
        # Jieba doesn't require training, but we need to build vocabulary
        vocab = set()
        tokenized_texts = []
        for text in texts:
            tokens = list(jieba.cut(str(text)))
            tokenized_texts.append(tokens)
            vocab.update(tokens)

        # Update vocabulary
        for word in vocab:
            if word not in self.vocab_to_index:
                self.vocab_to_index[word] = self.next_idx
                self.index_to_vocab[self.next_idx] = word
                self.next_idx += 1

        # Calculate max sequence length
        self.max_len = max(len(t) for t in tokenized_texts)

    def encode(self, text: str, max_len: Optional[int] = None) -> List[int]:
        tokens = list(jieba.cut(str(text)))
        indices = [self.vocab_to_index.get(t, 0) for t in tokens]
        pad_len = (max_len or self.max_len) - len(indices)
        return indices + [1] * pad_len if pad_len > 0 else indices[:max_len]


class SentencePieceTokenizer(Tokenizer):
    """SentencePiece subword tokenizer"""
    def __init__(self, model_prefix: str = "sp_model", vocab_size: int = 8000, model_dir: str = "Sentence Piece Models"):
        super().__init__()
        self.model_dir = os.path.normpath(model_dir)
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.abspath(os.path.join(self.model_dir, model_prefix))
        self.vocab_size = vocab_size
        self.sp = spm.SentencePieceProcessor()
        self.max_len = 0

    def train(self, texts: List[str]) -> None:
        # Save texts to temporary file
        tmp = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False)
        for text in texts:
            tmp.write(f"{text}\n")
        tmp_path = tmp.name
        tmp.close()

        # Train SentencePiece model
        spm.SentencePieceTrainer.train(
            input=tmp_path,
            model_prefix=self.model_path, # chinese dame
            vocab_size=self.vocab_size,
            pad_id=1,
            unk_id=0,
            bos_id=-1,
            eos_id=-1,
            model_type="unigram"
        )
        
        os.unlink(tmp_path)

        # Load trained model
        self.sp.load(f"{self.model_path}.model")
        
        # Build vocabulary mappings
        self.vocab_to_index = {self.sp.id_to_piece(i): i for i in range(self.sp.get_piece_size())}
        self.index_to_vocab = {i: p for p, i in self.vocab_to_index.items()}
        
        # Calculate max sequence length
        self.max_len = max(len(self.sp.encode_as_ids(t)) for t in texts)
        self.delete_temp_dir()

    def encode(self, text: str, max_len: Optional[int] = None) -> List[int]:
        ids = self.sp.encode_as_ids(text)
        pad_len = (max_len or self.max_len) - len(ids)
        return ids + [1] * pad_len if pad_len > 0 else ids[:max_len]
    
    def delete_temp_dir(self):
        shutil.rmtree(self.model_dir, ignore_errors=True)


def texts_to_index_sequences(
    texts: List[str],
    tokenizer: Tokenizer,
    train: bool = True
) -> Tuple[List[List[int]], Dict[str, int], Dict[int, str], int]:
    """Convert texts to index sequences using the specified tokenizer"""
    if train:
        tokenizer.train(texts)
    
    sequences = [tokenizer.encode(text, tokenizer.max_len) for text in texts]
    vocab, ivocab = tokenizer.get_vocab()
    return sequences, vocab, ivocab, tokenizer.max_len


def build_eval_sequences(
    texts: List[str],
    tokenizer: Tokenizer,
    text_len: int
) -> List[List[int]]:
    tokenizer.delete_temp_dir()
    """Convert evaluation texts to index sequences"""
    return [tokenizer.encode(text, text_len) for text in texts]