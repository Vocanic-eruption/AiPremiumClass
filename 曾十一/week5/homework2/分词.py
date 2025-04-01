import jieba

def segment_file(input_file_path, output_file_path):
    """
    对输入的 txt 文件进行分词，并将结果保存到输出文件中。
    
    :param input_file_path: 输入的 txt 文件路径
    :param output_file_path: 输出的 txt 文件路径
    """
    # 打开输入文件进行读取
    with open(input_file_path, "r", encoding="utf-8") as input_file:
        # 打开输出文件进行写入
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            # 逐行读取文件内容
            for line in input_file:
                # 使用 jieba 进行分词
                seg_list = jieba.cut(line.strip())
                # 将分词结果用空格连接并写入输出文件
                output_file.write(" ".join(seg_list) + "\n")

# 示例用法
if __name__ == "__main__":
    input_file_path = "/mnt/data_1/zfy/4/week5/homework_2/jianlai.txt"  # 输入文件路径
    output_file_path = "/mnt/data_1/zfy/4/week5/homework_2/jianlai_1.txt"  # 输出文件路径
    
    segment_file(input_file_path, output_file_path)
    print(f"分词完成，结果已保存到 {output_file_path}")





