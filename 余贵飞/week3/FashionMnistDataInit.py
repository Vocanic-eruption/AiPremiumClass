# encoding: utf-8
# @File  : FashionMnistDataInit.py
# @Author: GUIFEI
# @Desc : pytorch 神经网络数据准备
# @Date  :  2025/03/13
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms.v2 import ToTensor, Compose

import matplotlib.pyplot as plt

def down_load_train_data():
    '''
    下载FashionMNIST 训练数据集
    :return: 返回训练数据集
    '''
    fashion_mnist_train_data = datasets.FashionMNIST(
        root='../dataset/FashionMNIST',
        train=True, # 是否下载训练数据库集 true -> 下载训练集 false -> 下载测试集
        download=True, # 是否下载，true -> 下载数据，如果以及下载了，不会再下载
        transform=transforms.Compose(
            [transforms.v2.Resize((28, 28)),
             transforms.v2.ToImage(),
             transforms.v2.ToDtype(torch.float32, scale=True),
             transforms.Normalize((0.5,), (0.5,))
             ]
        ) # 主要作用是将图像数据转换为适合神经网络处理的张量格式 归一化处理‌ 自动将像素值范围从 0-255 线性缩放到 0.0-1.0 之间，具体通过除以255实现‌
    )
    return fashion_mnist_train_data

def down_load_test_data():
    '''
    下载FashionMNIST 测试数据集
    :return: 返回测试数据集
    '''
    fashion_mnist_test_data = datasets.FashionMNIST(
        root='../dataset/FashionMNIST',
        train=False, # 是否下载训练数据库集 true -> 下载训练集 false -> 下载测试集
        download=True, # 是否下载，true -> 下载数据，如果以及下载了，不会再下载
        transform=transforms.Compose(
            [transforms.v2.Resize((28, 28)),
             transforms.v2.ToImage(),
             transforms.v2.ToDtype(torch.float32, scale=True),
             transforms.Normalize((0.5,), (0.5,))
             ])
            # 主要作用是将图像数据转换为适合神经网络处理的张量格式 归一化处理‌ 自动将像素值范围从 0-255 线性缩放到 0.0-1.0 之间，具体通过除以255实现‌
    )
    return train_data


if __name__ == '__main__':
    # 加载训练数据集
    train_data = down_load_train_data()
    # 加载测试数据集
    test_data = down_load_test_data()
    print(train_data[1][0].shape) # torch.Size([1, 28, 28]) 一个颜色通道，其余两个值分别是图片的宽W 高H
    print(train_data[1][0].reshape(-1).shape) # torch.Size([784]) 把图片数据展平为一个一维张量
    label_set = set ([label for img, label in train_data])
    print(label_set)
    print(train_data[0][0])




