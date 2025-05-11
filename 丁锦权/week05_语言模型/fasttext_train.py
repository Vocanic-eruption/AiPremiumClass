import fasttext
import jieba

with open("week5/dpcq.txt",'r', encoding='utf-8') as f:
    read=f.read()
words=jieba.lcut(read)

with open("week5/dpcq_t.txt",'w', encoding='utf-8') as f:
    for word in words:
        f.write(word+' ')

model=fasttext.train_unsupervised("week5/dpcq_t.txt")

print(model.get_nearest_neighbors("萧炎"))
print(model.get_sentence_vector("美杜莎"))
print(model.get_word_vector("美杜莎"))
print(model.get_analogies('萧炎',"美杜莎",'小医仙'))
