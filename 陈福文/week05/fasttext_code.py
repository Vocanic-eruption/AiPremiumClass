import fasttext

model = fasttext.train_supervised('D:\\ai\\badou\\codes\\week05\\cooking.stackexchange.txt', lr=0.5, epoch=25, dim=50)

print(model.predict("Which baking dish is best to bake a banana bread ?"))