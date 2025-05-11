import jieba
import fasttext

# 文档预处理
# with open('xiyouji.txt', 'r', encoding='utf-8') as f:
#     lines = f.read()
    
# with open('xiyouji_cut.txt', 'w', encoding='utf-8') as f: # 分词后文件，用于训练词向量模型。
#     f.write(' '.join(jieba.cut(lines)))

model = fasttext.train_unsupervised('xiyouji_cut.txt',model = 'skipgram') # 训练模型，生成词向量。
print('文档词汇表长度：',len(model.words)) 

# 保存模型，用于后续加载使用。
model.save_model('xiyouji.bin') 

# 加载模型，用于后续使用。
model = fasttext.load_model('xiyouji.bin')


print(model.get_word_vector('猴哥')) # 查看词向量。

print(model.get_nearest_neighbors('猴哥', k=5)) # 查看相似词。

# 分析词间类比    猴哥-八戒=师傅-？  
print(model.get_analogies('猴哥', '八戒', '师傅'))


