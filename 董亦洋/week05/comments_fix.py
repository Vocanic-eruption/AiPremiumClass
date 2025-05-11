#修复后内容文件
fixed = open("douban_comments_fixed.txt","w",encoding="utf-8")

#修复前内容文件
lines = [line for line in open("doubanbook_top250_comments.txt","r",encoding="utf-8")]
prev_line = ""
for i, line in enumerate(lines):
    if i == 0:
        fixed.write(line.strip()+"\n")
        continue
    terms = line.split("\t")
    #是不是信息行
    if(len(terms) == 6):
        if(len(prev_line) != 0):
            fixed.write(prev_line+"\n")
        prev_line = line.strip()
    else:
        prev_line += line.strip()
        #最后一行
        if(i == len(lines)-1): fixed.write(prev_line)