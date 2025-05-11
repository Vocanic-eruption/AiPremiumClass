
# 修复后文件
fixed =  open('./data/comments_fixed.txt', 'w', encoding='utf-8')

# 读取修复前文件
lines = [line for line in open('./data/doubanbook_top250_comments.txt', 'r', encoding='utf-8')]

for i,line in enumerate(lines):

    # 目录
    if i == 0:
        fixed.write(line)
        continue
    
    prev_line = ''
    terms = line.split('\t')

    # 当前行的书名是否上一行的书名相同
    if terms[0] == prev_line.split('\t')[0]:
        if len(prev_line.split('\t')) == 6:
            fixed.write(prev_line+'\n')  
            prev_line = line.strip()  
        else:
            prev_line = ''
    else:
        if len(terms) == 6:
            fixed.write(line)           # 新书
            prev_line = line.strip()    
        else:
            prev_line += line.strip()   # 处理特殊情况
           

fixed.close()

        
    