import math
import unicodedata

from collections import Counter
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple, Optional

from utils.jieba_segmenter import JiebaSegmenter


class BM25Calculator:
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
    def compute_document_stats(
        doc_terms: Dict[str, List[str]]
    ) -> Tuple[Dict[str, int], float]:
        doc_lengths = {}
        total_length = 0
        
        for doc_name, terms in doc_terms.items():
            length = len(terms)
            doc_lengths[doc_name] = length
            total_length += length
            
        avgdl = total_length / len(doc_terms) if doc_terms else 0.0
        return doc_lengths, avgdl

    @staticmethod
    def compute_term_frequencies(
        doc_terms: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, int]]:
        doc_term_freq = {}
        
        for doc_name, terms in doc_terms.items():
            doc_term_freq[doc_name] = Counter(terms)
            
        return doc_term_freq

    @staticmethod
    def compute_idf(
        doc_term_freq: Dict[str, Dict[str, int]], 
        N: int
    ) -> Dict[str, float]:
        idf = {}
        unique_terms = set(term for term_freq in doc_term_freq.values() for term in term_freq)
        
        for term in unique_terms:
            doc_count = sum(1 for freq in doc_term_freq.values() if term in freq)
            idf[term] = math.log((N - doc_count + 0.5) / (doc_count + 0.5))
            
        return idf

    @staticmethod
    def compute_bm25(
        documents: Dict[str, List[str]],
        stopwords: Optional[List[str]] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> Tuple[csr_matrix, List[str], List[str]]:
        doc_terms, all_terms = BM25Calculator.tokenize_documents(documents, stopwords)
        doc_lengths, avgdl = BM25Calculator.compute_document_stats(doc_terms)
        doc_term_freq = BM25Calculator.compute_term_frequencies(doc_terms)
        idf = BM25Calculator.compute_idf(doc_term_freq, len(documents))
            
        unique_terms = list({term:1 for term in all_terms}.keys())
        vocab = {term:i for i, term in enumerate(unique_terms)}
        
        rows, cols, data = [], [], []
        for doc_idx, (doc_name, term_freq) in enumerate(doc_term_freq.items()):
            doc_len = doc_lengths[doc_name]
            for term, tf in term_freq.items():
                if term in vocab:
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
                    score = idf.get(term, 0.0) * (numerator / denominator)
                    
                    rows.append(doc_idx)
                    cols.append(vocab[term])
                    data.append(score)
        
        shape = (len(doc_term_freq), len(vocab))
        return csr_matrix((data, (rows, cols)), shape=shape), list(doc_term_freq.keys()), unique_terms
