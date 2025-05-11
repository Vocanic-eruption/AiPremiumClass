# 处理原始数据
lines = [line for line in open('./data/raw/doubanbook_top250_comments.txt', 'r', encoding='utf-8')]
with open('./data/fixed/doubanbook_top250_comments_fixed.txt', 'a', encoding='utf-8') as fixed:
    for i, line in enumerate(lines):
        line = line.strip()
        if line == '': continue
        terms = line.split('\t')
        if (len(terms) == 6):
            if i != 0: fixed.write('\n')
        else:
            fixed.write(' ')
        fixed.write(line)
