import fasttext

model = fasttext.train_supervised('cooking.stackexchange.txt', epoch=10, dim=200)

print(model.predict('How do I cover up the white spots on my cast iron stove?'))

model_WEL = fasttext.train_supervised('WELFake_Dataset_fixed.txt')

print(model_WEL.predict('Hillary’s crime family: End of days for the U.S.A'))
