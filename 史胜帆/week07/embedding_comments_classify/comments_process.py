# 对douban_movies_comments 进行数据预处理 提取出 ‘评论’ ‘投票’ 为后续文本分类做准备

import csv
import jieba
import matplotlib.pyplot as plt
import pickle

#用户评论数据集
ds_comments = []


#fixed = open('comments_processed.txt','w',encoding = 'utf-8')  不需要存下去

with open('douban_movies_comments.csv','r',encoding = 'utf-8') as f:
    reader = csv.DictReader(f,delimiter = ',')
    # 1 读取csv 取出对应项
    for line in reader:
        
        vote = int(line['votes']) # 严格整形 易于判断
        #去除0~5模糊不清的vote和其对应的comment 只保留 意向明显的项极好与极坏[0,5] 的vote和comment
        if vote in [0,5]:
            # 文本分类是基于tokenize（标记）的 因此同时进行jieba分词
            words = jieba.lcut(line['content'])
            # 把comment与 vote以元组的形式对应起来 再存入数据集中
            ds_comments.append((words,1 if vote == 5 else 0))
    
    # 2 看看到底有多少意向明了的电影评论 并观察数据集 看看有没有其他不良影响因素 需要继续数据预处理 便于后续数据分析
    print(len(ds_comments))
    print(ds_comments[:5]) # 取出前5条看看 发现可以在jieba中导入stop_words 避免标点符号对分类结果的影响
    # 发现有过短的评论数据 (['！'], 0)] 却给了差评 这种数据不具有普遍性 不利于我们的学习效果
    
    # 3 进一步数据清洗  去除 过长 过短 评论
    comments_len = [len(c) for c,v in ds_comments]
    # 观察评论分布情况 为后续过长过短评论的剔除作为选择依据
    # 用pyplot 画直方图观察分布情况
    # plt.hist(comments_len,bins = 50) # bins 把直方图的数值分布区间等分为等宽的bins份
    # plt.xlabel('长度')
    # plt.ylabel('数量')
    # plt.show() 
    # 用 pyplot 绘制箱形图 直接观察数据分布
    # plt.boxplot(comments_len)
    # plt.show()
    # 得到结论 去除 <10  >100 的评论

    ds_comments = [c for c in ds_comments if len(c[0]) in range(10,150)]
    # 筛选完毕
    

    # 对象数据持久化 pickle  包括存储对象的属性和方法等 也可存储其他数据
    # dump 对象->可存储数据（字节流）  load 可存储数据->对象
    with open('./comments.pkl','wb') as f:
        pickle.dump(ds_comments,f)
    # 这样就保存了 处理后的数据集合

    # 以上就是数据的处理和保存了

    # 后续 模型的搭建和模型的训练 在下一个专门的文件里完成
    