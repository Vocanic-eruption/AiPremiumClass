import os
os.chdir(os.path.dirname(__file__))  # 切换到 .py 文件所在目录
# 修复内容
fixed = open('doubanbook_top250_comments_fixed.txt', 'w', encoding='utf-8')
lines = open('doubanbook_top250_comments.txt', 'r', encoding='utf-8')
lines = [line for line in lines]
for i, line in enumerate(lines):
    if i == 0:
        fixed.write(line)
        prev_line = '' 
        continue
    terms = line.split('\t')
    
    if terms[0] == prev_line.split('\t')[0]:
        if len(prev_line.split('\t')) == 6:
           fixed.write(prev_line + '\n')
           prev_line = line.strip()
        else:
            prev_line = ""
    else:
        if len(terms) == 6:
            prev_line = line.strip()
        else:
            prev_line  += line.strip()
fixed.close()                