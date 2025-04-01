# 使用fasttext训练word2vec词向量模型，并计算词汇间的相关度

import os
import re
import jieba
import numpy as np
import pandas as pd
import fasttext
import fasttext.util
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors

# 设置文件路径
base_path = os.path.dirname(os.path.abspath(__file__))
douban_path = os.path.join(base_path, 'Douban_comments')
comments_file = os.path.join(douban_path, 'doubanbook_top250_comments.txt')

# 读取数据
def load_data():
    # 使用豆瓣评论作为自定义文档文本
    # 添加on_bad_lines='skip'参数跳过格式不正确的行
    comments_df = pd.read_csv(comments_file, sep='\t', on_bad_lines='skip')
    return comments_df

# 文本清洗
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 去除URL
    text = re.sub(r'http\S+', '', text)
    # 去除特殊字符和数字
    text = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 分词
def tokenize(text):
    text = clean_text(text)
    words = jieba.lcut(text)
    # 去除停用词（简单实现，实际应用中可以使用停用词表）
    words = [word for word in words if len(word) > 1]
    return words

# 准备训练数据
def prepare_training_data(comments_df):
    # 提取评论内容
    comments = comments_df['body'].fillna("").astype(str).tolist()
    print(f"加载了 {len(comments)} 条评论")
    
    # 清洗和分词
    tokenized_comments = [' '.join(tokenize(comment)) for comment in comments]
    print(f"分词后得到 {len(tokenized_comments)} 条处理后的评论")
    
    # 保存为训练文件 - 使用英文路径避免fasttext无法处理中文路径的问题
    # train_file = os.path.join(base_path, 'fasttext_train.txt')
    train_file = "E:/fasttext_train.txt"  # 使用固定的英文路径
    try:
        with open(train_file, 'w', encoding='utf-8') as f:
            count = 0
            for comment in tokenized_comments:
                if comment.strip():
                    f.write(comment + '\n')
                    count += 1
        print(f"成功写入 {count} 条评论到文件 {train_file}")
        
        # 验证文件是否可读
        if os.path.exists(train_file) and os.path.getsize(train_file) > 0:
            with open(train_file, 'r', encoding='utf-8') as f:
                first_lines = [next(f) for _ in range(3) if f]
                print(f"文件前几行内容预览: {first_lines}")
        else:
            print(f"警告: 文件 {train_file} 不存在或为空")
            
    except Exception as e:
        print(f"写入训练文件时出错: {e}")
        raise
    
    return train_file

# 训练词向量模型
def train_word2vec(train_file, model_file='fasttext_model.bin'):
    # 设置模型保存路径
    model_path = os.path.join(base_path, model_file)
    
    # 检查训练文件是否存在且可读
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"训练文件 {train_file} 不存在")
    
    if os.path.getsize(train_file) == 0:
        raise ValueError(f"训练文件 {train_file} 为空，无法训练模型")
    
    # 尝试打开文件进行读取测试
    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            sample_lines = [next(f) for _ in range(3) if f]
            print(f"训练文件前几行内容: {sample_lines}")
    except Exception as e:
        raise IOError(f"无法读取训练文件 {train_file}: {e}")
    
    print(f"开始训练模型，使用文件: {train_file}")
    print(f"文件大小: {os.path.getsize(train_file) / (1024*1024):.2f} MB")
    
    # 训练模型
    # skipgram模型，词向量维度100，窗口大小5，最小词频5
    try:
        model = fasttext.train_unsupervised(
            train_file, 
            model='skipgram',
            dim=100,
            epoch=5,
            lr=0.05,
            wordNgrams=1,
            minCount=5,
            bucket=200000,
            thread=4,
            ws=5
        )
        
        # 保存模型
        model.save_model(model_path)
        print(f"模型训练完成，已保存到 {model_path}")
        
        return model
    except Exception as e:
        print(f"训练模型时出错: {e}")
        # 尝试使用绝对路径
        abs_train_file = os.path.abspath(train_file)
        print(f"尝试使用绝对路径: {abs_train_file}")
        
        try:
            model = fasttext.train_unsupervised(
                abs_train_file, 
                model='skipgram',
                dim=100,
                epoch=5,
                lr=0.05,
                wordNgrams=1,
                minCount=5,
                bucket=200000,
                thread=4,
                ws=5
            )
            
            # 保存模型
            model.save_model(model_path)
            print(f"使用绝对路径成功训练模型，已保存到 {model_path}")
            
            return model
        except Exception as e2:
            raise RuntimeError(f"尝试使用绝对路径仍然失败: {e2}")

# 计算词汇相关度
def calculate_word_similarity(model, words):
    # 获取词向量
    word_vectors = {}
    for word in words:
        if word in model.words:
            word_vectors[word] = model.get_word_vector(word)
    
    # 计算词汇间的余弦相似度
    similarity_matrix = {}
    for word1 in word_vectors:
        similarity_matrix[word1] = {}
        vec1 = word_vectors[word1].reshape(1, -1)
        for word2 in word_vectors:
            if word1 != word2:
                vec2 = word_vectors[word2].reshape(1, -1)
                similarity = cosine_similarity(vec1, vec2)[0][0]
                similarity_matrix[word1][word2] = similarity
    
    return similarity_matrix

# 查找最相似的词
def find_most_similar_words(model, word, top_n=10):
    if word not in model.words:
        print(f"词汇 '{word}' 不在词汇表中")
        return []
    
    # 获取最相似的词
    word_vec = model.get_word_vector(word)
    similarities = []
    
    for w in model.words:
        if w != word:
            w_vec = model.get_word_vector(w)
            # 计算余弦相似度
            similarity = cosine_similarity(
                word_vec.reshape(1, -1), 
                w_vec.reshape(1, -1)
            )[0][0]
            similarities.append((w, similarity))
    
    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_n]

# 可视化词向量（使用PCA降维）
def visualize_word_vectors(model, words, output_file='word_vectors.png'):
    from sklearn.decomposition import PCA
    
    # 获取词向量
    word_vectors = []
    valid_words = []
    for word in words:
        if word in model.words:
            word_vectors.append(model.get_word_vector(word))
            valid_words.append(word)
    
    if len(valid_words) < 2:
        print("没有足够的有效词汇进行可视化")
        return
    
    # 使用PCA降维到2维
    pca = PCA(n_components=2)
    result = pca.fit_transform(word_vectors)
    
    # 可视化
    plt.figure(figsize=(12, 8))
    plt.scatter(result[:, 0], result[:, 1], c='blue', alpha=0.5)
    
    for i, word in enumerate(valid_words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]), fontsize=9)
    
    plt.title('Word Vectors Visualization')
    plt.savefig(os.path.join(base_path, output_file))
    plt.close()

# 导出为TensorBoard可视化格式（选做）
def export_for_tensorboard(model, output_dir='tensorboard_logs'):
    try:
        from tensorboardX import SummaryWriter
        
        # 创建输出目录
        output_path = os.path.join(base_path, output_dir)
        os.makedirs(output_path, exist_ok=True)
        
        # 创建SummaryWriter
        writer = SummaryWriter(output_path)
        
        # 获取所有词向量
        vectors = []
        meta = []
        for word in model.words:
            vectors.append(model.get_word_vector(word))
            meta.append(word)
        
        # 添加嵌入向量
        writer.add_embedding(np.array(vectors), metadata=meta)
        writer.close()
        
        print(f"词向量已导出到TensorBoard格式，保存在 {output_path}")
        print("使用以下命令启动TensorBoard：")
        print(f"tensorboard --logdir={output_path}")
    except ImportError:
        print("未安装tensorboardX，无法导出为TensorBoard格式")
        print("可以使用pip install tensorboardX安装")

# 主函数
def main():
    print("加载数据...")
    comments_df = load_data()
    
    print("准备训练数据...")
    train_file = prepare_training_data(comments_df)
    
    print("训练词向量模型...")
    model = train_word2vec(train_file)
    
    # 测试词汇相关度
    test_words = ['爱情', '生活', '文学', '历史', '科学', '哲学', '艺术', '人生', '社会', '政治']
    
    print("\n计算词汇相关度...")
    similarity_matrix = calculate_word_similarity(model, test_words)
    
    # 打印相似度矩阵
    print("\n词汇相关度矩阵:")
    for word1, similarities in similarity_matrix.items():
        print(f"\n与 '{word1}' 最相似的词:")
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
        for word2, sim in sorted_similarities:
            print(f"- {word2}: {sim:.4f}")
    
    # 查找与特定词最相似的词
    target_words = ['爱情', '生活', '文学']
    for word in target_words:
        print(f"\n与 '{word}' 最相似的10个词:")
        similar_words = find_most_similar_words(model, word)
        for w, sim in similar_words:
            print(f"- {w}: {sim:.4f}")
    
    # 可视化词向量
    print("\n可视化词向量...")
    visualize_word_vectors(model, test_words)
    
    # 导出为TensorBoard格式（选做）
    print("\n导出为TensorBoard格式...")
    export_for_tensorboard(model)

if __name__ == "__main__":
    main()