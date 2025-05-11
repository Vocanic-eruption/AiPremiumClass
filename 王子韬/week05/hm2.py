import fasttext
import jieba
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体，确保可以正确显示中文
try:
    font = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf')  # Windows
except:
    try:
        font = FontProperties(fname=r'/System/Library/Fonts/PingFang.ttc')  # Mac
    except:
        font = FontProperties()  # 使用默认字体

# 分词
def cut_words(file_path, file_name):
    with open(file_path, mode='r', encoding='utf-8') as f:
        text = f.read()
    
    # 分词，并保存到一个文件中
    with open(file_name, mode='w', encoding='utf-8') as f:
        f.write(' '.join(jieba.cut(text)))
    print('分词完成')


if __name__ == '__main__':
    novel_file = './水浒传.txt'
    segmented_file = './shuihuzhuan.txt'
    
    # 分词
    try:
        cut_words(novel_file, segmented_file)
    except FileNotFoundError:
        print(f"文件 '{novel_file}' 未找到。请确保文件存在并重试。")
        exit(1)
    
    # 使用skipgram模型
    model_type = 'skipgram'
    print(f'开始训练{model_type}模型...')
    
    # 训练word2vec模型
    try:
        model = fasttext.train_unsupervised(segmented_file, model=model_type,
                                        dim=100, epoch=10, lr=0.1, wordNgrams=2,
                                        loss='ns', bucket=200000, thread=4,
                                        minCount=1)
        
        # 保存模型
        model_file = 'shuihuzhuan.bin'
        model.save_model(model_file)
        print(f'模型已保存至 {model_file}')
        
        # 加载模型
        model = fasttext.load_model(model_file)
        
        # 词向量维度
        print('词向量的维度为', model.get_dimension())
        
        # 计算人物词汇间的相关度
        key_characters = ['宋江', '武松', '林冲', '鲁智深', '李逵']
        print('主要人物相似词:')
        for character in key_characters:
            print(f"\n与'{character}'最相似的词:")
            similar_words = model.get_nearest_neighbors(character, k=5)
            for score, word in similar_words:
                print(f"  {word}: {score:.4f}")
        
    except Exception as e:
        print(f"训练或处理过程中出错: {str(e)}")