"""
完整FastText词向量训练与可视化流程
"""

import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 
import os
import shutil # 用于删除文件
from torch.utils.tensorboard import SummaryWriter
from tensorboard.plugins import projector


BASE_DIR = "zfy68/week05/fasttext_test_data"
os.makedirs(BASE_DIR, exist_ok=True)

# 方法1：数据准备
def prepare_corpus():
    input_file = os.path.join(BASE_DIR, "doubanbook_top250_comments_fixed.txt")
    output_file = os.path.join(BASE_DIR, "corpus.txt")

    if os.path.exists(output_file):
        return output_file

    with open(input_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as w:
        for line in f:
            terms = line.split('\t')
            if len(terms) == 6:
                w.write(terms[5])

    print("语料库文件已生成:", output_file)
    return output_file


# 方法2：训练fasttext模型
def train_fasttext(corpus_path):
    model_file = os.path.join(BASE_DIR, "fasttext_model.bin")
    if os.path.exists(model_file):
        return fasttext.load_model(model_file)
    print("训练fasttext中...")
    try:
        model_result = fasttext.train_unsupervised(input=corpus_path,
                                                    model='skipgram',  # or 'cbow'
                                                    lr=0.05,  # 学习率
                                                    epoch=10,
                                                    wordNgrams=3)
        model_result.save_model(model_file)
    except Exception as e:
        print(f"训练失败: {e}")
        if not os.path.exists(corpus_path):
            print(f"错误：语料库文件不存在：{corpus_path}")
        else:
            print(f"错误：无法打开语料库文件：{corpus_path}")
        return None
    print(f"模型已保存到{model_file}")
    return model_result

# 方法3：词汇相似度计算  
def calculate_similarity(model):
    # 词对相似度    
    word_pairs = [("小说", "文学"), ("挺好", "真好"), ("还行", "一般")]
    print("词对相似度:")
    for word1, word2 in word_pairs:
        if word1 in model.words and word2 in model.words:  # 检查词汇是否存在
            vec1 = model.get_word_vector(word1).reshape(1, -1)
            vec2 = model.get_word_vector(word2).reshape(1, -1)
            cosine_sim = cosine_similarity(vec1, vec2)[0][0]
            print(f"{word1} - {word2}: {cosine_sim:.4f}")
        else:
            print(f"{word1} 或 {word2} 不在词汇表中")
    # 查找最近邻词汇
    target_words = ["小说","挺好","还行","爱"]
    print("最近邻词汇:")
    for word in target_words:
        try:
            print(f"与{word} 最相近的5个词:")
            neighbors = model.get_nearest_neighbors(word,k=5)
            print(f"{word}: {neighbors}")
        except:
            print(f"{word}: 无法计算")

# 方法4：保存词向量和元数据
def save_vectors_and_metadata(model, output_dir):
    vectors_file = os.path.join(output_dir, "vectors.tsv")
    metadata_file = os.path.join(output_dir, "metadata.tsv")

    with open(vectors_file, 'w', encoding='utf-8') as vectors, open(metadata_file, 'w', encoding='utf-8') as metadata:
        for word in model.words:
            vector = model.get_word_vector(word)
            # 保存词向量
            vectors.write('\t'.join([str(x) for x in vector]) + '\n')
            # 保存元数据
            metadata.write(word + '\n')
   
# 方法5：配置tensorboard 
def configure_tensorboard(output_dir, vectors_file, metadata_file):
    config = projector.ProjectorConfig() # 创建配置
    embedding = config.embeddings.add() # 添加嵌入
    embedding.tensor_name = "embedding"  # TensorBoard 中的嵌入名称
    embedding.metadata_path = metadata_file # 添加元数据文件路径
    embedding.tensor_path = vectors_file # 添加向量文件路径
    projector.visualize_embeddings(output_dir, config)


    

if __name__  == '__main__':

    # 1、数据准备
    corpus_path = prepare_corpus()

    # 2、训练fasttext模型
    model = train_fasttext(corpus_path)

    # 3、词汇相似度计算
    calculate_similarity(model)
    
    # 4、保存词向量和元数据
    save_vectors_and_metadata(model,
                              output_dir=BASE_DIR)
    
    # 5、配置tensorboard    
    configure_tensorboard(BASE_DIR, "vectors.tsv", "metadata.tsv")

   # 6、启动tensorboard
    writer = SummaryWriter(log_dir=BASE_DIR)
    writer.close()
    print("请在终端中运行以下命令：")
    print(f"tensorboard --logdir={BASE_DIR}")
    print("然后在浏览器中打开http://localhost:6006/查看结果")


    # 测试
    output_file = os.path.join(BASE_DIR, "corpus.txt")
    model1 = fasttext.train_unsupervised(output_file, model='skipgram')
    model2 = fasttext.train_unsupervised(output_file, model='cbow')

    neighbors1 = model1.get_nearest_neighbors("真好")
    neighbors2 = model2.get_nearest_neighbors("真好")
    print()
    print("neighbors1",neighbors1)
    print("neighbors2",neighbors2)


