# 读取豆瓣top250图书评论文件
lines = [line for line in open('./钟文迪/week05_语言模型及词向量/douban_comments.txt', 'r', encoding='utf-8')]
# print(lines[:5])

# 将处理结果保存到新的文件中
newFile = open("./钟文迪/week05_语言模型及词向量/process_douban_comments.txt", "w", encoding = 'utf-8')

for i, line in enumerate(lines):
    # 保存列标题
    if i == 0:
        newFile.write(line)
        prev_line = ''  # 将上一行的书名置为空
        continue

    # 提取书名和评论文本
    terms = line.split("\t")
    # print(terms)

    # 当前行的书名 == 上一行的书名
    if terms[0] == prev_line.split("\t")[0]:
        if len(prev_line.split("\t")) == 6:    # 上一行不是评论
            # 保存上一行记录
            newFile.write(prev_line + "\n")
            prev_line = line.strip()  # 保存当前行
        else:
            prev_line = ""
    else:
        if len(terms) == 6: # 新数评论
            prev_line = line.strip()
        else:
            prev_line += line.strip() # 合并当前行和上一行



newFile.close