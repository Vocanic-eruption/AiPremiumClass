import re
#修复后内容文件
fixed = open("WELFake_Dataset_fixed.txt","w",encoding="utf-8")
lines = [line for line in open('WELFake_Dataset.csv', 'r', encoding='utf-8')]

prev_line = ''
for i, line in enumerate(lines):
    #标题行抛弃
    if i == 0:
        continue

    #以[数字,]为开头的，视为新行，清空prev_line
    if re.match(r'^\d+,', line) is not None:
        prev_line = ''
    #不是[数字,]结尾，只保存内容
    if re.search(r',[01]{1}+$', line.strip()) is None:
        prev_line += re.sub(r'^\d+,','',line) #去头
    #以[0/1,]为结尾的，视为结束，最后数字为prev_line的标签
    if re.search(r',[01]{1}+$', line.strip()) is not None:
        temp = re.sub(r'^\d+,','',line) #去头
        temp = re.sub(r',\d+$','',temp) #去尾
        prev_line += temp
        # print(prev_line)
        label = re.search(r',\d+$', line).group().replace(',','')
        fixed.write('__label__'+ label +' ' + prev_line.replace('\n', '') + '\n')
