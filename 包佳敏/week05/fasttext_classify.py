import kagglehub

# Download latest version
#path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")

#print("Path to dataset files:", path)

import fasttext
import csv

#比pytorch快很多，因为底层是c++实现的

# word2vec模型使用方法，文本分类中一样可用

model = fasttext.train_supervised('cooking.stackexchange.txt')
#文本分类功能
print(model.predict('Which baking dish is best to bake a banana bread ?'))
print(model.predict('Why not put knives in the dishwasher?'))

to_fix = open('WELFake_Dataset.csv', 'r',encoding='utf-8')
fixed = open('WELFake_Dataset_fixed.txt', 'w',encoding='utf-8')    
lines = to_fix.readlines()
i = 0
previous_content = ''
previous_label = ''
for i, line in enumerate(lines):
    if i == 0:
        continue
    terms = line.split(",")
    content = terms[1:-1]
    content = ' '.join(content).strip()
    label = terms[-1].strip() 
    index = terms[0]
    if label != '0' and label != '1':
        previous_content = previous_content + ' ' + content
    else :
        if previous_content != '':
            fixed.write('__label__'+previous_label+' '+previous_content + "\n")
            previous_content = content
            previous_label = label 
        else:  
            previous_content = content
            previous_label = label     
            
to_fix.close()
fixed.close() 

model2 = fasttext.train_supervised('WELFake_Dataset_fixed.txt')
print(model2.predict('Hillary’s crime family: End of days for the U.S.A'))
