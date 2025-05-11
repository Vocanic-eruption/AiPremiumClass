import jieba

# fixed = open('./data/fixed/three_body_fixed.txt', 'w', encoding='utf-8')
# 加载停用词
stopwords = [line.strip() for line in open('./data/raw/stopwords.txt', 'r', encoding='utf-8')]

with open('./data/raw/three_body.txt', 'r', encoding='utf-8') as file, open('./data/fixed/three_body_fixed.txt', 'w', encoding='utf-8') as fixed:
    for line in file:
        line = line.strip()
        if line:
            cut_line = jieba.lcut(line)
            cut_line = [word for word in cut_line if word not in stopwords]
            fixed.write(' '.join(cut_line))