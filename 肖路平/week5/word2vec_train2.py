import jieba

with open('sgyy.txt', 'r', encoding='utf-8') as f:
    lines = f.read()

with open('sgyy_spr.txt', 'w', encoding='utf-8') as f:
    f.write('  '.join(jieba.cut(lines)))