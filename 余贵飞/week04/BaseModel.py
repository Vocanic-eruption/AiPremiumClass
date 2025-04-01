# encoding: utf-8
# @File  : BaseModel.py
# @Author: GUIFEI
# @Desc : olivetti faces 数据集模型基类定义
# @Date  :  2025/03/20
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import nn, overrides


class BaseModel(nn.Module):
    '''
    olivetti faces 数据集模型基类定义
    '''
    def __init__(
            self,
            input_dim = 4096,
            hidden_dims=None,
            norm_type = "batch",
            dropout_rate = 0.0,
            use_l2 = False,
            activate_func = "ReLU",
    ):
        '''
        :param input_dim: Olivetti Faces 输入特征维度 (64x64=4096)
        :param hidden_dims: 隐层维度
        :param norm_type:   归一化类型: "batch", "layer", "none"
        :param dropout_rate: 是否启用 L2 正则化 (通过优化器)
        :param use_l2: 是否使用L2
        :param activate_func:  # 默认激活函数,默认使用ReLU
        '''
        super(BaseModel, self).__init__()
        # 定义隐藏层神经元的数量
        if hidden_dims is None:
            # 如果不传递隐藏层神经元数量，使用默认值
            hidden_dims = [256, 128]
        self.norm_type = norm_type
        self.use_l2 = use_l2
        # 网络层级堆叠数组
        stock_layers = []

        # 动态构建网络层
        prev_dim = input_dim
        for i, dim in enumerate(hidden_dims):
            # 第一层隐藏层，输入特征数量 input_dim
            stock_layers.append(nn.Linear(prev_dim, dim))
            # 添加归一化层
            if norm_type == "batch":
                # norm_type 如果传入参数是 batch 使用 BatchNorm1d 归一化
                stock_layers.append(nn.BatchNorm1d(dim))
            elif norm_type == "layer":
                # norm_type 如果传入参数是 batch 使用 LayerNorm 归一化
                stock_layers.append(nn.LayerNorm(dim))

            # 添加激活函数，
            if activate_func == "ReLU":
                stock_layers.append(nn.ReLU())
            elif activate_func == "tanh":
                stock_layers.append(nn.Tanh())
            elif activate_func == "sigmoid":
                stock_layers.append(nn.Sigmoid())

            # 添加 Dropout
            if dropout_rate > 0:
                stock_layers.append(nn.Dropout(dropout_rate))
            # 将下一层隐藏层的输入特征复制本层隐藏层的输出
            prev_dim = dim

        # 输出层 (假设是分类任务)
        stock_layers.append(nn.Linear(prev_dim, 40))  # Olivetti 有 40 个类别
        # 构建神经网络模型
        self.model = nn.Sequential(*stock_layers)

    # 重写定义详情计算的方法
    def forward(self, x):
        return self.model(x)