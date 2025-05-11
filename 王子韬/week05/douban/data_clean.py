fixed =open("douban_comments_fixed.txt","w",encoding="utf-8")
#修复前内容文件
lines =[line for line in open("doubanbook_top250_comments.txt","r",encoding="utf-8")]

for i,line in enumerate(lines):
    #保存标题列
    if i==0:
        fixed.write(line)
        prev_line=''    #上一行的书名置为空
        continue
#提取书名和评论文本
    terms =line.split("\t")
#当前行的书名==上一行的书名
    if terms[0]==prev_line.split("\t")[0]:
        if len(prev_line.split("\t"))==6:#上一行是评论
            #保存上一行记录
           fixed.write(prev_line+'\n')
           prev_line =line.strip() #保存当前行
        else:   
           prev_line =""
    else:
        if len(terms)==6: #新书评论
            #fixed.write(line)
           prev_line =line.strip() #保存当前行
        else:
            prev_line +=line.strip()  #合并当前行和上一行
fixed.close()