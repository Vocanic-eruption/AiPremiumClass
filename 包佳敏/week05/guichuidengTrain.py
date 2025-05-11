import fasttext
import jieba 
with open('鬼吹灯.txt', 'r', encoding='utf-8') as file:
    lines = file.read()
with open('鬼吹灯_cut.txt', 'w', encoding='utf-8') as file:
    output = ' '.join(jieba.cut(lines))
    file.write(output)

model = fasttext.train_unsupervised('鬼吹灯_cut.txt', model='skipgram', lr=0.1, dim=100, ws=5, epoch=5, minCount=5, minn=3, maxn=6, neg=5, thread=12, t=1e-4, lrUpdateRate=100)  
print("文档词汇表",model.words) #打印词汇表  

#获取词向量
print(model.get_word_vector('胡国华')) #获取词向量

#获取近邻词
print(model.get_nearest_neighbors('胡国华', k=5)) #获取近邻词

#分析词间类比
print(model.get_analogies('胡国华', '舅舅', '孙先生')) #分析

#保存模型
model.save_model('鬼吹灯_c.bin') #保存模型

#加载模型
model = fasttext.load_model('鬼吹灯_c.bin') #加载模型