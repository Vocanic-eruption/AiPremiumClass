import math
import unicodedata

from scipy.sparse import csr_matrix
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

from .jieba_segmenter import JiebaSegmenter


class TFIDFCalculator:
    @staticmethod
    def tokenize_documents(
        documents: Dict[str, List[str]],
        stopwords: Optional[List[str]] = []
    ) -> Tuple[Dict[str, List[str]], List[str]]:
        def is_valid_token(tok: str) -> bool:
            for ch in tok:
                if ch in stopwords:
                    return False
                cat = unicodedata.category(ch)
                if not (cat.startswith('P')
                        or cat.startswith('Z')
                        or cat.startswith('C')):
                    return True
            return False

        doc_terms = {}
        all_terms = []
        
        for doc_name, doc_contents in documents.items():
            terms_for_doc = []
            for content in doc_contents:
                words = [word for word in JiebaSegmenter.cut(content) if is_valid_token(word)]
                terms_for_doc.extend(words)
                all_terms.extend(words)
            doc_terms[doc_name] = terms_for_doc
            
        return doc_terms, all_terms

    @staticmethod
    def compute_tf(tokenized_corpus: Dict[str, List[str]]) -> Dict[int, Dict[str, float]]:
        tf: Dict[int, Dict[str, float]] = {}
        for title, words in tokenized_corpus.items():
            counts: Dict[str, float] = defaultdict(float)
            for w in words:
                counts[w] += 1.0

            tf[title] = dict(counts)
        return tf

    @staticmethod
    def compute_idf(tokenized_corpus: Dict[str, List[str]], smooth: bool = True) -> Dict[str, float]:
        N = len(tokenized_corpus)
        df: Dict[str, int] = defaultdict(int)
        
        for _, words in tokenized_corpus.items():
            for w in set(words):
                df[w] += 1

        idf: Dict[str, float] = {}
        for w, df_count in df.items():
            if smooth:
                idf[w] = math.log((N + 1) / (df_count + 1))
            else:
                idf[w] = math.log(N / (df_count + 1))
        return idf

    @staticmethod
    def compute_tfidf(
        corpus: Dict[str, List[str]],
        stopwords: Optional[List[str]] = []
    ):
        tokenized, all_terms = TFIDFCalculator.tokenize_documents(corpus, stopwords)
        tf = TFIDFCalculator.compute_tf(tokenized)
        idf = TFIDFCalculator.compute_idf(tokenized)

        unique_terms = list({term:1 for term in all_terms}.keys())
        vocab = {term:i for i, term in enumerate(unique_terms)}
        
        rows, cols, data = [], [], []
        for doc_idx, (_, tf_vals) in enumerate(tf.items()):
            for term, tf_val in tf_vals.items():
                if term in vocab:
                    rows.append(doc_idx)
                    cols.append(vocab[term])
                    data.append(tf_val * idf.get(term, 0))
        
        return csr_matrix((data, (rows, cols)), shape=(len(tf), len(vocab))), list(tf.keys()), unique_terms
    
    @staticmethod
    def compute_tfidf_by_sklearn(
        corpus: Dict[str, List[str]],
        stopwords: Optional[List[str]] = []
    ):
        book_names = list(corpus.keys())

        processed_corpus = [
            ' '.join(' '.join(JiebaSegmenter.cut(comment)) for comment in comments)
            for comments in corpus.values()
        ]

        vec = TfidfVectorizer(stop_words=list(stopwords))
        tfidf = vec.fit_transform(processed_corpus)
        feature_names = vec.get_feature_names_out()
        
        return tfidf, book_names, feature_names