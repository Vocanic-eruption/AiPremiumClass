#############################豆瓣原始数据格式转换#################
#修复后内容存盘文件
fixed=open("douban_comment_fixed.txt","w",encoding="utf-8")

#修复前内容文件
lines=[line for line in open("doubanbook_top250_comments.txt","r",encoidng="utf-8")]

for i,line in enumerate(line):
    #保存标题列
    if i==0:
        fixed.write(line)
        prev_line=''
        continue
    #提取书名和评论文本
    terms=line.split("\t")

    #当前行的书名==上一行的书名
    if terms[0] ==prev_line.split("\t")==6:
        if len(prev_line.split("\t"))==6:
            fixed.write(prev_line+'\n')
            prev_line=line.strip()
        else:
        prev_line=""
    else:
        if len(terms)==6:
            prev_line=line.strip()
        else:
            prev_line+=line.strip()

fixed.close()


###############################计算TF-IDF并且通过余弦相似度给出推荐列表##########
import csv
import jieba
from sklearn.feature extraction.text import TfidfVectorizerfrom sklearn.metrics.pairwise import cosine similarityimport numpy as np
def load data(filename):
    # 图书评论信息集合
    book_comments ={} #{书名:“评论1词 + 评论2词 +..."}

    with open(filename,'r')as f:
        reader=csv.DictReader(f,delimiter='t')# 识别格式文本中标题列
        for item in reader:
            book = item['book']
            comment = item['body"]
            comment_words =jieba.lcut(comment)
            if book=='':continue #跳过空书名
            # 图书评论集合收集
            book_comments[book]=book comments.get(book,[])
            book_comments[book].extend(comment words)
    return book_comments

if __name__='main':
    # 加载停用词列表
    stop words = [line.strip() for line in open("stopwords.txt","r", encoding="utf-8")]
    
    # 加载图书评论信息book comments =load data("douban comments fixed.txt")
    print(len(book comments))

    #提取书名和评论文本
    book_names = []
    book_comms =[]
    for book,comments in book comments.items():
        book_names.append(book)
        book_comms.append(comments)

    # 构建TF-IDF特征矩阵
    vectorizer=TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix= vectorizer,fit_transform(['',join(comms)for comms in book_comms])
    
    # 计算图书之间的余弦相似度
    similarity_matrix=cosine_similarity(tfidf matrix)
    
    # 输入要推荐的图书名称
    book_list=list(book comments.keys())
    print(book list)
    book_name = input("请输入图书名称:")
    book_idx=book names.index(book name) #获取图书索引
    
    #获取与输入图书最相似的图书
    recommend book index= np.argsort(-similarity_matrix[book idx])[1:11]

    # 输出推荐的图书
    for idx in recommend book index:
        print(f"《{book names[idx]}》\t 相似度:{similarity matrix[book idx][idx]:.4f}")
    print()

##############FastText 实战############################
#######################无监督模型训练###########
import fasttext

#Skipgram model
model=fasttext.train_unsupervised('data.txt',model='skipgram')

#or,cbow model
model=fasttext.train_unsupervised('data.txt',model='cbow')

#获取词向量的最近邻，直观了解向量能够捕获的语义信息类型
model.get_nearest_neighbors('asparagus')

# 词汇间类比
model.get_analogies("berlin","germany","france" )


###################文本分类模型###############
import fasttext
model=fasttext.train_supervised('data.train.txt')
#检索单词与标签列表
print(model.words)
print(model.labels)


##########模型的保存与加载#############3
#save model
model.save_model("model_filename.bin")
#load model
model=FastText.load_model("model_filename.bin")



###############模型预测##########################
#返回预测概率最高的标签
model.predict("Which baking dish is best to bake a banana bread?")
#通过指定参数K来一簇额多个标签
model,predict("Which baking dish is best to bake a banana bread?",k=3)
#预测字符串数组
model.predict(["Which baking dish is best to bake a banana bread?","Why not put knives in the dishwasher?"],k=3)



###################量化压缩模型文件###############
#使用已训练的模型对象进行量化
model.quantize(input='data.trian.txt',retrain=True)
#显示结果并保存新模型
print_results(*model.test(valid_data))
model.save_model("model_filename.ftz")



############加载预训练的词向量文件###########
from fasttext import FastText
#fasttext预训练词向量文件
wv_file='path_to_your_vector_file'
#加载预训练的Fasttext词向量
word_vectors=FastText.load_model(wv_file)
#获取词汇表大小词向量维度
vocab_size=len(word_vectors.words)
embedding_dim=word_vectors.get_dimension()
#创建一个嵌入矩阵，每一个都是一个词的Fasttext
embedding_matrix=np.zeros((vocab_size,embedding_dim))
for i,word in enumerate(word_vectors.words):
        embedding_vector=word_vectors[word]
        if embedding_vector is not None:
                embedding_matrix[i]=embedding_vector
#在模型中使用预训练的Fasttext词向量
embedding=nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))


#############Tensorboaard词向量可视化############
import torch
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter()
meta=SummaryWriter()
meta=[]
while len(meta)<100:
    i=len(meta)
    meta=meta+word_vectors.words[i]
meta=meta[:100]

writer.add_embedding(embedding.weight[:100],metadata=meta)

