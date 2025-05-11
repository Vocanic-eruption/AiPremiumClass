### 【第五周作业】
#### 1. 实现基于豆瓣top250图书评论的简单推荐系统（TF-IDF及BM25两种算法实现）

+ 使用BM25模型实现基于豆瓣top250图书评论的简单推荐系统，见下文件
> douban_bm25.py

+ 使用TF-IDF模型实现基于豆瓣top250图书评论的简单推荐系统，见下文件
> douban_tfidf.py

```text
过程中遇到了哪些问题?
1. 书名和评论的提取，中间涉及到类型转换，此处用了较多的时间去挨个了解数据提取方式，还遇到了很多编码格式错误的问题。
2. 使用模型后，发现推荐的值 与实际输入值关系不大，甚至出现了0.001这样的相似度，于是自己去了解了模型参数值，不断调整。
```
#### 2. 使用自定义的文档文本，通过fasttext训练word2vec训练词向量模型，并计算词汇间的相关度。（选做：尝试tensorboard绘制词向量可视化图）

> hlm_train.py
```text
过程中遇到了哪些问题?
模型构建的时候，由于参数不正确，导致界面处于灰度状态，一直卡流程
```

#### 3. 使用课堂示例cooking.stackexchange.txt，使用fasttext训练文本分类模型。（选做：尝试使用Kaggle中的Fake News数据集训练文本分类模型）
https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification