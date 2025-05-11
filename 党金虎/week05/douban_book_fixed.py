from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
from tqdm import tqdm

# 读取数据
w = open('./党金虎/week05/douban_test_data/doubanbook_top250_comments_fixed.txt', 'w', encoding='utf-8')

f = open('./党金虎/week05/douban_test_data/doubanbook_top250_comments.txt', 'r', encoding='utf-8')
lines = f.readlines()
f.close()
print(len(lines))
print(lines[0])

for i, line in enumerate(tqdm(lines)):
    if i == 0:
        w.write(line)
        prev_line = ''
        continue
    terms = line.split('\t')
    prev_terms = prev_line.split('\t')

    # 判断书名是否一样
    if terms[0] == prev_terms[0]:
        # 书名一样,长度为6 完整的一行
        if len(prev_terms) == 6:
            # 写入上一行
            w.write(prev_line + '\n')
            prev_line = line.strip()
        else:
            # 异常情况
            prev_line = ''
    else:
        # 书名不一样,判断当前行是否为完整的一行
        if len(terms) == 6: 
            prev_line = line.strip() 
        else: 
            # 如果不完整,则合并到上一行
            prev_line+= line.strip() 


w.close