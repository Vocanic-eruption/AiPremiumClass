import jieba


def text_process():
    text = open('diy_text.txt', 'r', encoding='utf-8')

    ai_file = open('ai.txt', 'w', encoding='utf-8')
    lines = [line for line in text]
    for line in lines:
        if line == '\n':
            continue
        words = jieba.lcut(line)
        ai_file.write(' '.join(words))
    ai_file.close()

def comments_clean():
    fixed_file = open('./comments_fixed.txt', 'w', encoding='utf-8')

    orgFile = open('./doubanbook_top250_comments.txt', 'r', encoding='utf-8')

    lines = [line for line in orgFile]

    """
        数据清洗思路：读下一条的时候保存上一条的数据
    """
    pre_content = ''
    for index, current_content in enumerate(lines):
        if index == 0:
            # 第一行直接记录
            print(repr(current_content))
            fixed_file.write(current_content)
            continue
        # 快速查看字符串中转义字符的可见形式
        terms = current_content.split("\t")
        if len(terms) == 6:
            # 正常数据
            if pre_content:
                fixed_file.write(pre_content + "\n")
            pre_content = current_content.strip()
        else:
            pre_content = pre_content + current_content.strip()
    fixed_file.close()


if __name__ == '__main__':
    text_process()
