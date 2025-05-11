import fasttext
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
import torch.nn as nn


def train_model():
    model = fasttext.train_supervised('cooking.stackexchange.txt', dim=200, epoch=20, lr=0.05)
    print(model)
    model.save_model('model.bin')


def main():
    model = fasttext.load_model('model.bin')
    # print(model.words)
    print(model.get_analogies('bread', 'butter', 'soup'))
    # print(model.get_word_vector('healthy'))
    print(model.get_nearest_neighbors('healthy'))
    print(model.predict('How to process a banana bread?', k=3))


    vocab_size = len(model.words)
    embedding_dim = model.get_dimension()
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for i, word in enumerate(model.words):
        embedding_vector = model[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))

    writer = SummaryWriter()
    meta = []
    while len(meta) < 100:
        i = len(meta)
        meta = meta + model.words[:i]
    meta = meta[:100]

    writer.add_embedding(embedding.weight[:100], metadata=meta)
    writer.close()


if __name__ == '__main__':
    # train_model()
    main()