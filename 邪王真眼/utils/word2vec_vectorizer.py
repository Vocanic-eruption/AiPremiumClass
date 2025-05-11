import torch
import numpy as np

from typing import List, Tuple
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel

from utils.jieba_segmenter import JiebaSegmenter


GENSIM_WORD2VEC_WEIGHTS_PATH = './邪王真眼/models_pretrained/model.w2v'
CORPUS_PATH = './邪王真眼/datasets/word2vec/corpus.txt'


class GensimVectorizer:
    def __init__(self, use_pretrained=True, dim=512):
        if use_pretrained:
            self.model = Word2Vec.load(GENSIM_WORD2VEC_WEIGHTS_PATH)
        else:
            sentences = []
            with open(CORPUS_PATH, encoding="utf8") as f:
                for line in f:
                    sentences.append(JiebaSegmenter.cut(line))
            self.model = Word2Vec(sentences, vector_size=dim, sg=1)
        self.vector_size = self.model.vector_size

    def word_to_vector(self, word: str) -> np.ndarray:
        try:
            return self.model.wv[word]
        except KeyError:
            return np.zeros(self.vector_size)

    def text_to_vector(self, text: str) -> np.ndarray:
        words = JiebaSegmenter.cut(text)
        if not words:
            return np.zeros(self.vector_size)
        vectors = [self.word_to_vector(w) for w in words]
        return np.mean(vectors, axis=0)

    def batch_texts_to_vectors(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self.text_to_vector(t) for t in texts])


# embedding = 768
class BERTVectorizer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        pretrained_model_name = 'bert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.bert.to(self.device)
        self.bert.eval()
        self.vector_size = self.bert.config.hidden_size
        self.vocab_size = self.tokenizer.vocab_size

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def text_to_vector(self,
                       text: str,
                       pooling: str = 'cls') -> np.ndarray:
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state

        if pooling == 'cls':
            vec = last_hidden[:, 0, :]
        elif pooling == 'mean':
            mask = attention_mask.unsqueeze(-1)
            sum_vec = (last_hidden * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            vec = sum_vec / lengths
        else:
            raise ValueError("Unsupported pooling type: choose 'cls' or 'mean'.")

        return vec.squeeze(0).cpu().numpy()

    def batch_texts_to_vectors(self,
                               texts: List[str],
                               pooling: str = 'cls') -> np.ndarray:
        return np.vstack([self.text_to_vector(t, pooling) for t in texts])

    def text_to_sequence(self,
                         text: str,
                         max_length: int = None,
                         padding: str = 'max_length') -> Tuple[torch.Tensor, torch.Tensor]:

        inputs = self.tokenizer(
            text,
            padding=padding,
            max_length=max_length,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state

        return hidden_states.squeeze(0).cpu(), attention_mask.squeeze(0).cpu()

    def text_to_indices(self,
                    text: str,
                    max_length: int = None,
                    padding: str = 'max_length') -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(
            text,
            padding=padding,
            max_length=max_length,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        return input_ids, attention_mask