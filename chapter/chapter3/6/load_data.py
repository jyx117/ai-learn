# -*- coding = utf-8 -*-
"""
softmax回归的从零开始实现
@project: ai-learn
@Author：michael
@file： load_data.py
@date：2023/4/10 4:07 下午
"""

import torch
import torchvision
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from d2l import torch as d2l
import numpy as np
import pandas as pd

d2l.use_svg_display()

batch_size = 256


def get_dataloader_workers():  # @save
    """使用4个进程来读取数据"""
    return 4


def load_data_fashion_mnist(batch_size, resize=None):  # @save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


class FMDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.images = df.iloc[:, 1:].values.astype(np.uint8)
        self.labels = df.iloc[:, 0].values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28, 1)
        label = int(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.tensor(image / 255., dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return image, label


def load_data_by_file():
    image_size = 28
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        # 这一步取决于后续的数据读取方式，如果使用内置数据集读取方式则不需要
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    train_df = pd.read_csv("./data/fashion-mnist_train.csv")
    test_df = pd.read_csv("./data/fashion-mnist_test.csv")
    train_data = FMDataset(train_df, data_transform)
    test_data = FMDataset(test_df, data_transform)
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True),
        DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    );
