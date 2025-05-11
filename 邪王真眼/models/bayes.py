import math

from utils.jieba_segmenter import JiebaSegmenter


class BayesApproach:
    def __init__(self, dataset):
        word_freq = {}
        label_count = {}
        self.word_count = 0
        
        v = set()
        for batch_data in dataset:
            label = batch_data.pop("label")
            text = batch_data.pop("x")
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1

            words = JiebaSegmenter.cut(text)
            v.update(words)
            for word in words:
                self.word_count += 1

                if label not in word_freq:
                    word_freq[label] = {}
                
                if word not in word_freq[label]:
                    word_freq[label][word] = 1
                    continue
                
                word_freq[label][word] += 1

        
        self.p_label = {label : count / sum(label_count.values()) for label, count in label_count.items()}
        self.p_word = {}
        for label, word_count in word_freq.items():
            total_count = sum(word_count.values())
            self.p_word[label] = {word : (1 + count) / (total_count + len(v)) for word, count in word_count.items()}
            self.p_word[label]['<unk>'] = 1 / (total_count + len(v))

    def forward(self, x):
        words = JiebaSegmenter.cut(x)
        result = {}

        for label, prob_label in self.p_label.items():
            log_prob = math.log(prob_label)
            for word in words:
                word_prob = self.p_word[label].get(word, self.p_word[label]['<unk>'])
                log_prob += math.log(word_prob)
            result[label] = log_prob

        max_log = max(result.values())
        exp_sum = sum(math.exp(val - max_log) for val in result.values())
        result = {label: math.exp(val - max_log) / exp_sum for label, val in result.items()}
        return {'out': result}
    
    def __call__(self, x):
        return self.forward(x)