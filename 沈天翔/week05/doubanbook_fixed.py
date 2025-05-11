import os
from tqdm import tqdm

fixed_file = 'doubanbook_top250_comments_fixed.txt'

def doubanbook_fixed():

    # # 检查文件是否存在
    # if os.path.exists(fixed_file_path):
    #     print(f"文件 {fixed_file_path} 已经存在。")
    #     return

    # 打开写入文件
    with open(fixed_file, 'w', encoding='utf-8') as fixed:
        # 打开读取文件
        with open('doubanbook_top250_comments.txt', 'r', encoding='utf-8') as f:
            # 逐行读取文件
            lines = f.readlines()

            # 逐行处理文件
            for i, line in enumerate(tqdm(lines)):
                # 保存标题列
                if i == 0:
                    fixed.write(line)
                    prev_line = ''  # 上一行的书名为置空
                    continue

                # 提取书名和评论文本
                terms = line.split('\t')

                # 当前书名 == 上一行书名
                if terms[0] == prev_line.split('\t')[0]:
                    if len(prev_line.split('\t')) == 6:
                        fixed.write(prev_line)  # 保存上一行
                        prev_line = line  # 保存当前行
                # 当前书名 != 上一行书名
                    else:
                        prev_line = ''  # 上一行的书名为置空
                else:
                    if len(terms) == 6:  # 当前行是评论
                        prev_line = line.strip()  # 更新上一行
                    else:
                        prev_line += line.strip()  # 合并当前行和上一行

# 运行程序
if __name__ == '__main__':
    doubanbook_fixed()
