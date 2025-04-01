import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import matplotlib as plt


# 加载数据集
def load_data():
    train_data = FashionMNIST(root="F:/githubProject/AiPremiumClass/梁锐江/week3/fashion_data", train=True,
                              download=True, transform=ToTensor())
    test_data = FashionMNIST(root="F:/githubProject/AiPremiumClass/梁锐江/week3/fashion_data", train=False,
                             download=True, transform=ToTensor())
    return train_data, test_data



if __name__ == '__main__':
    tr_data = load_data()[0]
    print(tr_data[1])
