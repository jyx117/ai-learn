# -*- coding = utf-8 -*-
"""
softmax回归的从零开始实现
@project: ai-learn
@Author：michael
@file： load_data.py
@date：2023/4/10 4:07 下午
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd


def get_dataloader_workers():
    return 4


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


def load_data_by_file(batch_size, train_file_path, test_file_path):
    image_size = 28
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        # 这一步取决于后续的数据读取方式，如果使用内置数据集读取方式则不需要
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)
    train_data = FMDataset(train_df, data_transform)
    test_data = FMDataset(test_df, data_transform)
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=get_dataloader_workers(),
                   drop_last=True),
        DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=get_dataloader_workers())
    )


if __name__ == '__main__':
    load_data_by_file(256)
