import csv
import jieba

def load_comments(file_name):
    """
    从指定文件中加载图书评论，并将评论分词后存储在字典中。

    参数:
    file_name (str): 包含图书评论的文件路径。

    返回:
    dict: 键为图书名称，值为评论分词后的列表。
    """
    book_comments = {}

    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t') # 识别格式文本中的标题列
        for item in reader:
            book = item.get('book', '').strip()
            comment = item.get('body', '').strip()

            if not book: continue

            comments_words = jieba.lcut(comment)

            book_comments[book] = book_comments.get(book, [])
            book_comments[book].extend(comments_words)
            # print(f'{book}')

    keys_to_remove = [key for key, value in book_comments.items() if len(value) < 150]
    for key in keys_to_remove:
        del book_comments[key]

    return book_comments

if __name__ == '__main__':
    file_name = 'doubanbook_top250_comments_fixed.txt'
    book_comments = load_comments(file_name)
    # print(f'{book_comments.keys()}')
    # print(f'{len(book_comments)}')
