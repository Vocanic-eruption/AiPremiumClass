import math
import unicodedata

from collections import defaultdict
from typing import Dict, List


class NewWordsDetector:
    def __init__(self, max_word_length: int = 5):
        self.max_word_length = max_word_length
        self.word_count: Dict[str, int] = defaultdict(int)
        self.left_neighbors: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.right_neighbors: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.total_count_by_length: Dict[int, int] = {}
        self.pmi: Dict[str, float] = {}
        self.left_entropy: Dict[str, float] = {}
        self.right_entropy: Dict[str, float] = {}
        self.scores: Dict[str, float] = {}

    def load_corpus(self, path: str) -> None:
        with open(path, encoding="utf8") as f:
            for line in f:
                sentence = line.strip()
                length = len(sentence)
                for n in range(1, self.max_word_length + 1):
                    for i in range(length - n + 1):
                        token = sentence[i:i + n]
                        self.word_count[token] += 1
                        if i > 0:
                            self.left_neighbors[token][sentence[i - 1]] += 1
                        if i + n < length:
                            self.right_neighbors[token][sentence[i + n]] += 1

    @staticmethod
    def _calc_entropy(neighbor_counts: Dict[str, int]) -> float:
        total = sum(neighbor_counts.values())

        if total <= 1:
            return 0.0
        
        entropy = 0.0
        for cnt in neighbor_counts.values():
            p = cnt / total
            entropy -= p * math.log(p)

        return entropy

    def _calc_total_counts(self) -> None:
        self.total_count_by_length = defaultdict(int)
        for token, cnt in self.word_count.items():
            self.total_count_by_length[len(token)] += cnt

    def compute_pmi(self) -> None:
        self._calc_total_counts()
        unigram_total = self.total_count_by_length.get(1, 1)
        for token, cnt in self.word_count.items():
            length = len(token)

            if length < 2:
                continue

            log_p_token = math.log(cnt) - math.log(self.total_count_by_length[length])
            
            log_p_chars = 0.0
            for ch in token:
                char_count = self.word_count.get(ch, 1)
                log_p_chars += math.log(char_count) - math.log(unigram_total)
            
            self.pmi[token] = (log_p_token - log_p_chars) / length

    def compute_entropy(self) -> None:
        for token in self.word_count:
            self.left_entropy[token] = self._calc_entropy(self.left_neighbors[token])
            self.right_entropy[token] = self._calc_entropy(self.right_neighbors[token])

    def compute_scores(self) -> None:
        for token, pmi_val in self.pmi.items():
            if len(token) < 2 or any(unicodedata.category(ch).startswith('P') for ch in token):
                continue
            le = self.left_entropy.get(token, 0.0)
            re = self.right_entropy.get(token, 0.0)
            self.scores[token] = pmi_val * min(le, re)

    def detect(self, corpus_path: str) -> List[str]:
        self.load_corpus(corpus_path)
        self.compute_pmi()
        self.compute_entropy()
        self.compute_scores()

        return sorted(self.scores, key=self.scores.get, reverse=True)


if __name__ == "__main__":
    detector = NewWordsDetector(max_word_length=5)
    results = detector.detect("./邪王真眼/datasets/new_words_detect/sample_corpus.txt")
    
    for length in (2, 3, 4):
        top = [w for w in results if len(w) == length][:10]
        print(f"Top-{length}", top)
        