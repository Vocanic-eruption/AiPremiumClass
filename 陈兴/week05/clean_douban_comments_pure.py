import re

def clean_comments_pure(input_file, output_file):
    """
    使用纯Python清洗豆瓣评论数据，将多行评论合并为一行
    """
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 初始化变量
    cleaned_lines = []
    current_record = None
    header = lines[0]  # 保存表头
    
    # 处理每一行
    for i in range(1, len(lines)):
        line = lines[i]
        # 检查是否是新记录的开始（包含6个tab分隔的字段）
        if len(line.split('\t')) >= 6:
            # 如果有当前记录，保存它
            if current_record:
                cleaned_lines.append(current_record)
            # 开始新记录
            current_record = line.strip()
        else:
            # 将当前行添加到当前记录中
            if current_record:
                # 替换换行为空格
                current_record += ' ' + line.strip()
    
    # 添加最后一条记录
    if current_record:
        cleaned_lines.append(current_record)
    
    # 写入清洗后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header)  # 写入表头
        for line in cleaned_lines:
            f.write(line + '\n')
    
    print(f"数据清洗完成，共处理 {len(cleaned_lines)} 条评论")
    print(f"清洗后的数据已保存至: {output_file}")

# 文件路径
input_file = '/Users/chenxing/AI/AiPremiumClass/陈兴/week05/doubanbook_top250_comments.txt'
output_file = '/Users/chenxing/AI/AiPremiumClass/陈兴/week05/doubanbook_top250_comments_single_line.txt'

# 执行清洗
clean_comments_pure(input_file, output_file)