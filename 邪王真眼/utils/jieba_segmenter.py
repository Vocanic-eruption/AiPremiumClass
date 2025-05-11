import jieba

from typing import List


class JiebaSegmenter:
    @staticmethod
    def cut(text: str) -> List[str]:
        return list(jieba.cut(text))


if __name__ == '__main__':
    text = "测试字符串"
    print(JiebaSegmenter.cut(text))
