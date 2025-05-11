import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import json

from typing import List, Dict, Optional, Tuple

from utils.jieba_segmenter import JiebaSegmenter
from utils.tfidf_calculator import TFIDFCalculator


class TfidfSummarizer:
    def load_data(self, file_path: str) -> Tuple[Dict[int, Dict[str, float]], List[str]]:
        with open(file_path, encoding="utf8") as f:
            documents = json.load(f)

        corpus = {}
        for doc in documents:
            title = doc["title"].replace("\n", " ")
            content = doc["content"].replace("\n", " ")
            corpus[title] = [content]

        tfidf_matrix, book_names, feature_names  = TFIDFCalculator.compute_tfidf_by_sklearn(corpus)

        tfidf_dicts = []
        for row in tfidf_matrix:
            row_dict = {
                feature_names[i]: row[0, i]
                for i in row.nonzero()[1]
            }
            tfidf_dicts.append(row_dict)

        indexed_corpus = {
            idx: (title, corpus[title][0]) for idx, title in enumerate(book_names)
        }

        return tfidf_dicts, indexed_corpus

    @staticmethod
    def generate_document_abstract(
        document_tf_idf: Dict[str, float],
        content: str,
        top: int = 3
    ) -> Optional[str]:
        sentences = [s.strip() for s in re.split(r"[？！？。]", content) if s.strip()]
        if len(sentences) <= 5:
            return None

        scored: List[tuple] = []
        for idx, sent in enumerate(sentences):
            words = JiebaSegmenter.cut(sent)
            score = sum(document_tf_idf.get(w, 0.0) for w in words) / (len(words) + 1)
            scored.append((score, idx))

        top_idxs = sorted(idx for _, idx in sorted(scored, reverse=True)[:top])
        abstract = "。".join(sentences[i] for i in top_idxs)
        return abstract

    def summarize(
        self,
        json_path: str,
        top: int = 3
    ) -> List[Dict[str, str]]:
        tfidf_list, corpus = self.load_data(json_path)
        results: List[Dict[str, str]] = []

        for doc_idx, tfidf in enumerate(tfidf_list):
            title, content = corpus[doc_idx]
            abstract = self.generate_document_abstract(tfidf, content, top=top)
            if abstract:
                results.append({
                    "标题": title,
                    "正文": content,
                    "摘要": abstract
                })

        return results


if __name__ == "__main__":
    summarizer = TfidfSummarizer()
    summaries = summarizer.summarize("./邪王真眼/datasets/news/news.json", top=3)

    with open("./邪王真眼/datasets/news/abstract.json", "w", encoding="utf8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"已生成 {len(summaries)} 篇文章摘要，输出到 abstract.json")
