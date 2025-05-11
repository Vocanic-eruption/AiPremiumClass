import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json

from typing import List, Tuple, Dict

from utils.tfidf_calculator import TFIDFCalculator


class TfidfSearchEngine:
    def __init__(self, json_path: str, top_n: int = 3):
        self.top_n = top_n
        self.tf_idf_dict, self.corpus = self._load_data(json_path)

    def _load_data(self, file_path: str) -> Tuple[Dict[int, Dict[str, float]], List[str]]:
        with open(file_path, encoding="utf8") as f:
            documents = json.load(f)
        corpus = [doc["title"] + "\n" + doc["content"] for doc in documents]

        tfidf_input = {f"doc_{i}": [content] for i, content in enumerate(corpus)}
        tfidf_matrix, _, feature_names = TFIDFCalculator.compute_tfidf_by_sklearn(tfidf_input)

        tfidf_dicts = []
        for i in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix[i]
            row_dict = {feature_names[j]: row[0, j] for j in row.nonzero()[1]}
            tfidf_dicts.append(row_dict)

        return tfidf_dicts, corpus

    def search(self, query: str) -> List[Tuple[int, float]]:
        words = TFIDFCalculator.tokenize_documents({"query": [query]})[0]["query"]

        scores = [
            (doc_id, sum(self.tf_idf_dict[doc_id].get(w, 0.0) for w in words))
            for doc_id in range(len(self.tf_idf_dict))
        ]
        results = sorted(scores, key=lambda x: x[1], reverse=True)[: self.top_n]

        for doc_id, score in results:
            print(f"得分: {score:.4f}")
            print(self.corpus[doc_id])
            print("--------------")
        return results


if __name__ == "__main__":
    engine = TfidfSearchEngine("./邪王真眼/datasets/news/news.json", top_n=3)
    while True:
        q = input("请输入您要搜索的内容: ")
        engine.search(q)
