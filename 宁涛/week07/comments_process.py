import csv
import jieba
import matplotlib.pyplot as plt
import pickle



# 用户评论数据集
ds_comments = []

# 1. Read the CSV file
with open('DMSC.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        vote = int(row['Star'])
        if vote in [1, 2, 4, 5]:
            words = jieba.lcut(row['Comment'])   
            ds_comments.append((words, 1 if vote in [1, 2] else 0))  # 1 for positive, 0 for negative

print(len(ds_comments))

comments_len = [len(c) for c,v in ds_comments]
plt.boxplot(comments_len)
plt.show

ds_comments = [c for c in ds_comments if len(c[0]) in range(60, 120)]
comments_len = [len(c) for c,v in ds_comments]
print(len(comments_len))

with open('comments.pkl', 'wb') as f:
    pickle.dump(ds_comments, f)
