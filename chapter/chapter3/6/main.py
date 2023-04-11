# -*- coding = utf-8 -*-
"""
@project: ai-learn
@Author：michael
@file： main.py
@date：2023/4/10 4:13 下午
"""
from d2l import torch as d2l
from load_data import load_data_by_file
import torch
from chapter.animator import Animator
from chapter.accumulator import Accumulator


def softmax(X):
    X_exp = torch.exp(X)
    print('X_exp:', X_exp.shape)
    partition = X_exp.sum(1, keepdim=True)
    print('partition:', partition.shape)
    value = X_exp / partition
    print('value:', value.shape)
    return value  # 这里应用了广播机制


# 神经网络
def net(X):
    t = X.reshape((-1, W.shape[0]))
    print('net:', t.shape, W.shape[0])
    t1 = torch.matmul(X.reshape((-1, W.shape[0])), W)
    print('t1:', t1.shape)
    res = torch.matmul(X.reshape((-1, W.shape[0])), W) + b
    print('res:', res.shape)
    return softmax(res)


# 损失函数：交叉熵
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):  # @save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):  # @save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def updater(batch_size):
    """ 优化器 """
    return d2l.sgd([W, b], lr, batch_size)


def train_epoch_ch3(net, train_iter, loss, updater):  # @save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        print('X:', X.shape, ', y:', y.shape)
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    print('metric:', metric[0] / metric[2], ', ', metric[1] / metric[2])
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  # @save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    print('train_loss:', train_loss, ", train_acc:", train_acc)


def predict_ch3(net, test_iter, n=6):  # @save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


if __name__ == '__main__':
    # 数据迭代器
    batch_size = 256
    train_iter, test_iter = load_data_by_file(batch_size)

    # 初始化W和b，图片是28*28的矩阵，拉伸为784的向量，输出为10个分类，因此
    num_inputs = 784
    num_outputs = 10
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    print('w:', W.shape)
    b = torch.zeros(num_outputs, requires_grad=True)
    print('b:', b.shape)

    lr = 0.1
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

    predict_ch3(net, test_iter)
    d2l.plt.show()
