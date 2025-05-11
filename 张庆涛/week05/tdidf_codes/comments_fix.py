

# 修复后内容存盘文件
fixed = open('douban_comments_fixed.txt','w',encoding='utf-8')

# 修复前内容文件
lines = [ line  for line in open('doubanbook_top250_comments.txt','r',encoding='utf-8')]

for index,line in enumerate(lines):
  # 保存标题列
  if(index == 0):
    fixed.write(line)
    prev_line = '' # 上一行的书名空
    continue
  # 去除空行
  # 提取书名和对应的评论文本 
  terms = line.split("\t") 
  # 当前行的书名 == 上一行的书名
  if terms[0] == prev_line.split("\t")[0]:
    if len(prev_line.split("\t")) == 6:
      # 保存上一行的数据
      fixed.write(prev_line + '\n')
      prev_line = line.strip() # 保存当前行
    else:
      prev_line = ""  
  else:
    if len(terms) == 6:
      # 记录是一本新书
      # fixed.write(line)
      prev_line = line.strip() # 保存当前行
    else:
      prev_line += line.strip()   # 合并上一行和当前行 

fixed.close() 