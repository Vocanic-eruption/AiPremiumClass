import pandas as pd
import re
import jieba
from collections import Counter

# 1. 数据加载与标签处理
df = pd.read_csv('doubanmovieshortcomments.csv')
df = df[df['Score'].isin([1,2,4,5])] 
df['label'] = df['Score'].apply(lambda x: 1 if x in [1,2] else 0)  

# 2. 文本清洗函数
def clean_text(text):
    text = re.sub(r'[^\u4e00-\u9fa5]', '', str(text)) 
    return text.strip()

# 3. 使用jieba分词
def tokenize_jieba(text):
    return list(jieba.cut(text))

# 应用预处理
df['cleaned'] = df['Comment'].apply(clean_text)
df['tokens'] = df['cleaned'].apply(tokenize_jieba)

# 4. 构建词典
all_tokens = [token for tokens in df['tokens'] for token in tokens]  
vocab_counter = Counter(all_tokens)
vocab = ['<PAD>', '<UNK>'] + [word for word, cnt in vocab_counter.most_common(10000)]  
word2idx = {word:i for i, word in enumerate(vocab)} 

print(f"词典大小: {len(vocab)}")
print("示例词汇:", vocab[:10])
