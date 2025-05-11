# encoding: utf-8
# @File  : BookDataProcess.py
# @Author: GUIFEI
# @Desc : 豆瓣书评数据解析处理
# @Date  :  2025/03/26

def data_process(original_data_path, fixed_data_path):
    """
    书评数据处理
    :param original_data_path: 数据处理后保存路径
    :param fixed_data_path: 原始数据路径
    :return: None
    """
    # 读取文件原始数据
    fixed = open(fixed_data_path, 'w', encoding='utf-8')
    lines = [line for line in open(original_data_path, 'r', encoding='utf-8')]
    pre_line = ''
    for i, line in enumerate(lines):
        if i == 0:
            pre_line = ''
            fixed.write(line)
            continue
        if line.strip() == '':
            continue
        elements = line.split('\t')
        # 取出书名
        pre_line_elements = pre_line.split('\t')
        if pre_line == '':
            if len(elements) == 6:
                pre_line = line.strip()
        else:
            if line.split('\t')[0] == pre_line_elements[0]:
                # 如果当前行的书名与上一行的书名相同
                if len(pre_line_elements) == 6:
                    # 将上一行的数据写入文件
                    fixed.write(pre_line + '\n')
                    pre_line = line.strip()
                else:
                    pre_line = ''
            else:
                if len(elements) == 6:
                    pre_line = line.strip()
                    fixed.write(pre_line + '\n')
                else:
                    pre_line += line.strip()
    fixed.write(pre_line + '\n')
    fixed.close()


if __name__ == '__main__':
    fixed_data_path = '../dataset/douban/doubanbook_top250_comments_fixed.txt'
    original_data_path = "../dataset/douban/doubanbook_top250_comments.txt"
    # 处理并保存处理后数据
    data_process(original_data_path, fixed_data_path)




