import pandas as pd
import jieba
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Any
from tqdm import tqdm
import re
import sentencepiece as spm


class Config:
    CSV_PATH = Path(r"D:\work\code\practice\home_work\杨文博\week07\DMSC.csv")
    JIEBA_DATA_PATH = Path(r"D:\work\code\practice\home_work\杨文博\week07\data\jieba.pkl")  # 使用.pkl扩展名更明确
    SP_PATH = Path(r"D:\work\code\practice\home_work\杨文博\week07\data\SentencePiece.pkl")
    MIN_WORDS = 10
    MAX_WORDS = 150

    MAX_LENGTH = 100  # 最大序列长度（超过将被截断）
    MIN_FREQ = 3  # 词汇最小出现频率

    # 模型超参数
    EMBEDDING_DIM = 16  # 词向量维度
    HIDDEN_SIZE = 128  # RNN隐藏层维度
    NUM_LAYERS = 1  # RNN层数
    DROPOUT = 0.5  # Dropout概率

    # 训练配置
    BATCH_SIZE = 64  # 批量大小
    LEARNING_RATE = 0.001  # 学习率
    NUM_EPOCHS = 20  # 训练轮数
    EARLY_STOPPING = 3  # 早停轮数（验证集损失不下降时）

    # 其他配置
    SAVE_MODEL = True  # 是否保存模型
    PRINT_EVERY = 50  # 每隔多少批次打印一次训练信息
    SEED = 42  # 随机种子


class CommentClassifier(ABC):
    @staticmethod
    @abstractmethod
    def classify(data: pd.DataFrame) -> List[Tuple[List[str], int]]:
        """
        抽象分类方法
        :param data: 包含评论和评分的DataFrame
        :return: 返回分词结果和对应标签的列表
        """
        pass


class JiebaClassifier:
    path = Config.JIEBA_DATA_PATH

    @staticmethod
    def clean_text(text: str) -> str:
        """去除标点符号和空格"""
        # 匹配所有非中文字符、非字母、非数字的字符（包括标点和空格）
        return re.sub(r'[^\w\u4e00-\u9fff]+', '', text)

    @staticmethod
    def classify(data: pd.DataFrame) -> List[Tuple[List[str], int]]:
        # 预处理：过滤空值并转换类型
        clean_data = data[["Comment", "Star"]].dropna()
        comments = clean_data["Comment"].astype(str).tolist()
        stars = clean_data["Star"].astype(int).tolist()

        # 批量清洗和分词（使用tqdm显示进度）
        word_lists = []
        for comment in tqdm(comments, desc="清洗和分词进度"):
            # 先清洗文本
            cleaned_text = JiebaClassifier.clean_text(comment)
            # 再分词
            words = jieba.lcut(cleaned_text)
            # 去除分词后可能存在的空字符串
            words = [word for word in words if word.strip()]
            word_lists.append(words)

        # 向量化处理标签
        labels = [0 if star < 4 else 1 for star in stars]

        # 组合并过滤结果
        return [
            (words, label)
            for words, label in zip(word_lists, labels)
            if Config.MIN_WORDS <= len(words) <= Config.MAX_WORDS
        ]


class SentencePieceClassifier:
    path = Config.SP_PATH
    modelPath = "spm.model"
    vocabSize = 8000  # 可调整

    @staticmethod
    def train_sentencepiece_model(comments: List[str], model_prefix: str = "spm", vocab_size: int = 8000):
        """训练 SentencePiece 模型"""
        with open("spm_input.txt", "w", encoding="utf-8") as f:
            for comment in comments:
                f.write(comment + "\n")
        spm.SentencePieceTrainer.train(
            input="spm_input.txt",
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=0.9995,  # 中文推荐设置高一点
            model_type="unigram"  # 也可以选择 bpe、char、word
        )

    @staticmethod
    def clean_text(text: str) -> str:
        """去除标点符号和空格"""
        return re.sub(r'[^\w\u4e00-\u9fff]+', '', text)

    @staticmethod
    def classify(data: pd.DataFrame) -> List[Tuple[List[str], int]]:
        clean_data = data[["Comment", "Star"]].dropna()
        comments = clean_data["Comment"].astype(str).tolist()
        stars = clean_data["Star"].astype(int).tolist()

        # 清洗文本
        cleaned_comments = [SentencePieceClassifier.clean_text(c) for c in comments]

        # 如果模型不存在就训练一个
        import os
        if not os.path.exists(SentencePieceClassifier.modelPath):
            SentencePieceClassifier.train_sentencepiece_model(cleaned_comments)

        # 加载模型
        sp = spm.SentencePieceProcessor()
        sp.load(SentencePieceClassifier.modelPath)

        # 编码
        word_lists = []
        for text in tqdm(cleaned_comments, desc="SentencePiece 分词进度"):
            pieces = sp.encode(text, out_type=str)
            word_lists.append(pieces)

        # 标签转换
        labels = [0 if star < 4 else 1 for star in stars]

        # 过滤结果
        return [
            (words, label)
            for words, label in zip(word_lists, labels)
            if Config.MIN_WORDS <= len(words) <= Config.MAX_WORDS
        ]

class DataProcessor:
    def __init__(self):
        self.data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """加载数据并进行基本清洗"""
        if not Config.CSV_PATH.exists():
            raise FileNotFoundError(f"CSV文件不存在: {Config.CSV_PATH}")

        data = pd.read_csv(Config.CSV_PATH)
        # 基本数据清洗
        data = data.dropna(subset=["Comment", "Star"])
        return data

    def process_and_save(self, classifier: CommentClassifier) -> None:
        """
        处理数据并保存结果
        :param classifier: 评论分类器实例
        """
        processed_data = classifier.classify(self.data)

        with open(classifier.path, "wb") as f:
            pickle.dump(processed_data, f)

        print(f"数据处理完成，已保存到: {Config.JIEBA_DATA_PATH}")
        print(f"处理后的数据量: {len(processed_data)}")


if __name__ == '__main__':
    try:
        processor = DataProcessor()
        classifier = JiebaClassifier()
        classifier2 = SentencePieceClassifier()
        processor.process_and_save(classifier2)
    except Exception as e:
        print(f"程序运行出错: {e}")
