from typing import Dict, List, Set, Optional


class MaximumForwardSegmenter:
    DICT_PATH = './邪王真眼/datasets/cut_words/dict.txt'

    _word_set: Optional[Set[str]] = None
    _prefix_dict: Optional[Dict[str, int]] = None
    _prefix_rev: Optional[Dict[str, int]] = None
    _max_len: Optional[int] = None

    @classmethod
    def _init_cache(cls):
        if cls._word_set is None:
            with open(cls.DICT_PATH, encoding='utf8') as f:
                cls._word_set = {line.strip().split()[0] for line in f}
            cls._max_len = max((len(w) for w in cls._word_set), default=1)
    
    @classmethod
    def _init_prefix_dict(cls):
        if cls._prefix_dict is None and cls._word_set:
            cls._prefix_dict = {}
            for w in cls._word_set:
                for i in range(1, len(w)):
                    prefix = w[:i]
                    if prefix not in cls._prefix_dict:
                        cls._prefix_dict[prefix] = 0
                cls._prefix_dict[w] = 1

    @classmethod
    def _init_prefix_rev(cls):
        if cls._prefix_rev is None and cls._word_set:
            cls._prefix_rev = {}
            for w in cls._word_set:
                rev_w = w[::-1]
                for i in range(1, len(rev_w)):
                    pre = rev_w[:i]
                    if pre not in cls._prefix_rev:
                        cls._prefix_rev[pre] = 0
                cls._prefix_rev[rev_w] = 1

    @classmethod
    def get_word_set(cls) -> Set[str]:
        cls._init_cache()
        return cls._word_set

    @classmethod
    def compute_max_len(cls) -> int:
        cls._init_cache()
        return cls._max_len

    @classmethod
    def max_forward_cut1(cls, text: str) -> List[str]:
        cls._init_cache()
        segments: List[str] = []
        idx = 0
        n = len(text)
        while idx < n:
            length = min(cls._max_len, n - idx)
            candidate = text[idx:idx + length]
            while candidate not in cls._word_set and len(candidate) > 1:
                candidate = candidate[:-1]
            segments.append(candidate)
            idx += len(candidate)
        return segments

    @classmethod
    def max_forward_cut2(cls, text: str) -> List[str]:
        cls._init_cache()
        cls._init_prefix_dict()
        
        segments: List[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = start + 1
            last_match = text[start:start+1]
            while end <= n and text[start:end] in cls._prefix_dict:
                if cls._prefix_dict[text[start:end]] == 1:
                    last_match = text[start:end]
                end += 1
            segments.append(last_match)
            start += len(last_match)
        return segments

    @classmethod
    def max_backward_cut1(cls, text: str) -> List[str]:
        cls._init_cache()
        segments: List[str] = []
        idx = len(text)
        while idx > 0:
            length = min(cls._max_len, idx)
            candidate = text[idx - length:idx]
            while candidate not in cls._word_set and len(candidate) > 1:
                candidate = candidate[1:]
            segments.insert(0, candidate)
            idx -= len(candidate)
        return segments

    @classmethod
    def max_backward_cut2(cls, text: str) -> List[str]:
        cls._init_cache()
        cls._init_prefix_rev()
        
        rev_text = text[::-1]
        segments_rev: List[str] = []
        start, n = 0, len(rev_text)
        while start < n:
            end = start + 1
            last_match = rev_text[start:start+1]
            while end <= n and rev_text[start:end] in cls._prefix_rev:
                if cls._prefix_rev[rev_text[start:end]] == 1:
                    last_match = rev_text[start:end]
                end += 1
            segments_rev.append(last_match)
            start += len(last_match)
        return [seg[::-1] for seg in reversed(segments_rev)]
    
    
if __name__ == "__main__":
    text = "测试字符串"
    print('Forward1:', MaximumForwardSegmenter.max_forward_cut1(text))
    print('Forward2:', MaximumForwardSegmenter.max_forward_cut2(text))
    print('Backward1:', MaximumForwardSegmenter.max_backward_cut1(text))
    print('Backward2:', MaximumForwardSegmenter.max_backward_cut2(text))
