# -*- coding = utf-8 -*-
"""
softmax回归简洁实现
@project: ai-learn
@Author：michael
@file： main.py
@date：2023/4/11 1:47 下午
"""

from d2l import torch as d2l
from chapter.load_data import load_data_by_file
import torch
from torch import nn


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
net.apply(init_weights)

batch_size = 256
train_file_path = '../../data/fashion-mnist_train.csv'
test_file_path = '../../data/fashion-mnist_test.csv'
train_iter, test_iter = load_data_by_file(batch_size, train_file_path, test_file_path)

# 损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
