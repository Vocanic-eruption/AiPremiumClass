txt = "D:\\ai\\badou\\codes\\第五周\\doubanbook_top250_comments.txt"
fixedTxt = "D:/ai/badou/codes/第五周/fixed_comments.txt"

fixed = open(fixedTxt,"w",encoding="utf-8")

readComments = open(txt,"r",encoding="utf-8")

lines = [line for line in readComments]

for i,line in enumerate(lines):
    if i == 0:
        fixed.write(line)
        prev_line = '' 
        continue
    terms = line.split("\t")
    if terms[0] == prev_line.split("\t")[0]:
        if len(prev_line.split("\t")) == 6:
            #保存上一行记录
            fixed.write(prev_line + '\n' )
            prev_line = line.strip()
        else:
            prev_line = "";
    else:
        if len(terms) == 6: 
            prev_line = line.strip()
        else:
            prev_line += line.strip()
fixed.close()
readComments.close()