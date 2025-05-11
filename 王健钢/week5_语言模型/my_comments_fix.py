file_path1 = r"./王健钢\week5_语言模型\douban_comments_fixed.txt"
#保存整理后的文本
fixed = open(file_path1, "w", encoding="utf-8")
#读取文件
file_path2 = r"./王健钢\week5_语言模型\doubanbook_top250_comments.txt"
lines = [line for line in open(file_path2, "r", encoding="utf-8")]
#遍历文件
for i,line in enumerate(lines):
    #保存标题
    if i == 0:
        fixed.write(line)
        prev_line = ''
        continue
    #保存评论
    terms = line.split("\t")

    if terms[0] == prev_line.split("\t")[0]:
        if len(prev_line.split("\t")) == 6:
            fixed.write(prev_line + '\n')
            prev_line = line.strip() #暂存当前行
        else:
            prev_line = ''
    else:
        if len(terms) == 6: #新书的评论
            prev_line = line.strip()
        else:
            prev_line += line.strip()

            
fixed.close()