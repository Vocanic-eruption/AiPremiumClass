## RNN 文本分类
### douban_movies_comments 数据预处理 包括 1去除模糊不清 意向不明确 对模型训练效果造成模糊影响的评论 2去除过短过长评论
### embedding 文本->索引->词向量
### PAD UNK 
### pad_sequence + DataLoader.callate_fn 实现 在不同batch中不同词向量长度的对齐
### 使用豆瓣电影评论数据集 做分本分类 分类电影评论的 positive / negative
