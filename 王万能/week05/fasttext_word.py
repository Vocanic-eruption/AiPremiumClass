import fasttext_patch
# import fasttext

model = fasttext_patch.train_supervised('cooking.stackexchange.txt',epoch = 20,dim = 200)
print(model.predict('How much does potato starch affect a cheese'))
print(model.predict('Regulation and balancing of readymade packed mayonnaise and other sauces'))
print(model.predict('How long can batter sit before chemical'))