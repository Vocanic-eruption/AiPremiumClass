import fasttext
import jieba
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def data_etl():
    with open('week04\\fasttext\\sanguo_parse.txt', 'w', encoding='utf-8') as fw:
        with open('week04\\files\\sanguo.txt', 'r', encoding='utf-8') as fr:
            for line in fr:
                if not line:
                    continue
                words = jieba.cut(line.strip())
                fw.write(' '.join(words)+'\n')

if __name__ == '__main__':
    # 数据预处理
    # data_etl()
    # 训练模型
    # model = fasttext.train_unsupervised('week04\\fasttext\\sanguo_parse.txt',model='cbow')
    # 保存模型
    # model.save_model('week04\\fasttext\\sanguo_model.bin')
    
    # 加载模型
    model = fasttext.load_model('week04\\fasttext\\sanguo_model.bin')
    
    # 创建 TensorBoard writer
    writer = SummaryWriter('week04/runs/sanguo_model_embeddings')
    

    # 获取与"关羽"相似的词及其向量
    similar_chars = model.get_nearest_neighbors('关羽', k=5)
    similar_vectors = []
    similar_names = ['关羽']  # 包含原词
    similar_vectors.append(model.get_word_vector('关羽'))  # 包含原词向量
    
    for score, name in similar_chars:
        similar_names.append(name)
        similar_vectors.append(model.get_word_vector(name))
    similar_vectors = np.array(similar_vectors)
    
    # 添加相似词向量到 TensorBoard
    writer.add_embedding(similar_vectors, metadata=similar_names, tag='similar_to_guanyu')
    writer.close()
    
    # 打印结果
    print(model.get_nearest_neighbors('关羽', k=3))
    print('-'*20)
    print(model.get_analogies('关羽','赵云','刘备', k=3))