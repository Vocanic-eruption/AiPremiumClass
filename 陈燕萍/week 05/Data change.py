from tqdm import tqdm

with open("C:/Users/stephanie.chen/miniconda3/envs/py312/doubanbook_top250_comments.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    
fixed = open("C:/Users/stephanie.chen/miniconda3/envs/py312/doubanbook_comments_fixed.txt", "w", encoding="utf-8")
lines = [line for line in open("C:/Users/stephanie.chen/miniconda3/envs/py312/doubanbook_top250_comments.txt", "r", encoding="utf-8")]

cached_line = ''
for i, line in enumerate(lines):
    if i == 0:
        fixed.write(line)  # 写入表头
        continue

    terms = line.split("\t")
    prev_terms = cached_line.split("\t") if cached_line else []

    if cached_line and terms[0] == prev_terms[0]:  # 书名相同
        if len(prev_terms) == 6:
            fixed.write(cached_line + '\n')  # 上一条完整，写入
            cached_line = line.strip()       # 当前行暂存
        else:
            cached_line = ''  # 上一条异常，跳过
    else:
        if len(terms) == 6:
            cached_line = line.strip()  # 新的完整行
        else:
            cached_line += line.strip()  # 拼接上一行未完的内容

# 处理最后一条
if cached_line and len(cached_line.split("\t")) == 6:
    fixed.write(cached_line + '\n')

fixed.close()

