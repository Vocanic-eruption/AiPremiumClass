import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np

from enum import Enum

from utils.cal_editing_distance import cal_editing_distance
from utils.cal_jaccard_distance import cal_jaccard_distance
from utils.jieba_segmenter import JiebaSegmenter
from utils.word2vec_vectorizer import BERTVectorizer
from utils.bm25_calculator import BM25Calculator


class AlgoType(Enum):
    EDITING = 'editing'
    JACCARD = "jaccard"
    BM25 = "bm25"
    WORD2VEC = "word2vec"


class QASystem:
    def __init__(self, know_base_path, algo):
        self.load_know_base(know_base_path)
        
        self.algo = algo
        if algo == AlgoType.WORD2VEC:
            self.vectorizer = BERTVectorizer()
            self.target_to_vectors = {}
            for target, questions in self.target_to_questions.items():
                vectors = self.vectorizer.batch_texts_to_vectors(questions)
                self.target_to_vectors[target] = np.array(vectors)
        else:
            pass

    def load_know_base(self, know_base_path):
        self.target_to_questions = {}
        with open(know_base_path, encoding="utf8") as f:
            for _, line in enumerate(f):
                content = json.loads(line)
                questions = content["questions"]
                target = content["target"]
                self.target_to_questions[target] = questions
        return

    def query(self, query):
        results = []
        if self.algo == AlgoType.EDITING:
            for target, questions in self.target_to_questions.items():
                scores = [cal_editing_distance(query, question) for question in questions]
                score = max(scores)
                results.append([target, score])
        elif self.algo == AlgoType.JACCARD:
            for target, questions in self.target_to_questions.items():
                scores = [cal_jaccard_distance(query, question) for question in questions]
                score = max(scores)
                results.append([target, score])
        elif self.algo == AlgoType.BM25:
            bm25, _ = BM25Calculator.compute_bm25(self.target_to_questions)
            words = JiebaSegmenter.cut(query)
            for doc, dict in bm25.items():
                score = 0
                for word in words:
                    if word in dict:
                        score += dict[word]
                results.append([doc, score])
        elif self.algo == AlgoType.WORD2VEC:
            query_vector = self.vectorizer.text_to_vector(query)
            for target, vectors in self.target_to_vectors.items():
                query_vector_norm = query_vector / np.linalg.norm(query_vector)
                vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
                cos = query_vector_norm.dot(vectors_norm.T)
                results.append([target, np.mean(cos)])
        else:
            assert "unknown algorithm!!"
        sort_results = sorted(results, key=lambda x:x[1], reverse=True)
        return sort_results


if __name__ == '__main__':
    qa_system = QASystem("./邪王真眼/datasets/corpus_p8/train.json", AlgoType.BM25) # 中文可能跑不了
    question = "话费是否包月超了"
    result = qa_system.query(question)
    print(result)
