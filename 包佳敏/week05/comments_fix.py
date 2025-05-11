fixed = open('comments_fixed.txt', 'w',encoding='utf-8')    

lines = [line for line in open('/Users/baojiamin/Downloads/doubanbook_top250_comments.txt', 'r',encoding='utf-8')]

print(len(lines))

last_line = ''
for i, line in enumerate(lines):
    #保存标题列
    if i == 0:
        fixed.write(line.strip())
        previous = ''
        continue
    #保存数据列
    terms = line.split("\t")
    if len(terms) == 6 and previous.split("\t")[0] == terms[0]: #新的一行同书名，保存
        fixed.write(previous + "\n")
        previous = line.strip() #去掉首尾空格
    else:
       if len(terms) != 6: #上一行的评论没结束，继续添加
           previous += line.strip()
       else: #新的一行书名不同，保存
           fixed.write(previous + "\n")
           previous = line.strip() #去掉首尾空格
    last_line = previous 

fixed.write(last_line + "\n")    
fixed.close()      