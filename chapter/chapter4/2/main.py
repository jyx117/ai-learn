# -*- coding = utf-8 -*-
"""
多层感知机复杂实现
@project: ai-learn
@Author：michael
@file： main.py
@date：2023/4/11 3:55 下午
"""

from chapter.load_data import load_data_by_file
import torch
from torch import nn
from d2l import torch as d2l


def relu(X):
    """激活函数"""
    a = torch.zeros_like(X)
    return torch.max(X, a)


def net(X):
    """模型"""
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)  # 这里“@”代表矩阵乘法
    H = relu(H @ W2 + b2)
    H = relu(H @ W3 + b3)
    H = (H @ W4 + b4)
    return H


if __name__ == '__main__':
    batch_size = 256
    train_file_path = '../../data/fashion-mnist_train.csv'
    test_file_path = '../../data/fashion-mnist_test.csv'
    train_iter, test_iter = load_data_by_file(batch_size, train_file_path, test_file_path)

    # 模型参数
    num_inputs, num_outputs, num_hiddens, num_hiddens2, num_hiddens3 = 784, 10, 256, 128, 64
    W1 = nn.Parameter(torch.randn(
        num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(
        num_hiddens, num_hiddens2, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_hiddens2, requires_grad=True))
    W3 = nn.Parameter(torch.randn(
        num_hiddens2, num_hiddens3, requires_grad=True) * 0.01)
    b3 = nn.Parameter(torch.zeros(num_hiddens3, requires_grad=True))
    W4 = nn.Parameter(torch.randn(
        num_hiddens3, num_outputs, requires_grad=True) * 0.01)
    b4 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    params = [W1, b1, W2, b2, W3, b3, W4, b4]

    # 损失函数
    loss = nn.CrossEntropyLoss(reduction='none')

    num_epochs, lr = 20, 0.1
    updater = torch.optim.SGD(params, lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
