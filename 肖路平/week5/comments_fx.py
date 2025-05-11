#修复后文件
fixed = open("douban_comments_fixed.txt","w",encoding="utf-8")

#修复前文件
lines = open("doubanbook_top250_comments.txt","r",encoding="utf-8").readlines()

for i,line in enumerate(lines):
    #保持标题列
    if i == 0:
        fixed.write(line)
        prev_line = ''
        continue
    #提取书名和评论
    terms = line.split("\t")

    #当前行的书名== 上一行的书名
    if terms[0] == prev_line.split("\t")[0]: #两种情况，可能都是评论，可能都是标题
        if len(prev_line.split("\t"))==6:   #上一行是书名
            fixed.write(prev_line + "\n")
            prev_line += line.strip() #保存当前行
        else: #上一行是评论,那就不用记录
            prev_line =""
    else: #如果不同，那么也有两种情况，可能一个书名，一个是评论
        if len(terms)==6: #当前行是书名，上一行是评论            
            prev_line=line.strip()
        else:#当前行是评论，上一行也是评论，那就进行合并
            prev_line +=line.strip() 
fixed.close()