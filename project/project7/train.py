import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import datetime


# 加载数据集，自己重写DataSet类
class MyDataset(Dataset):
    # root_dir为数据目录，label_file，为标签文件
    def __init__(self, root_dir, label_file, transform=None):
        self.root_dir = root_dir
        self.label = np.loadtxt(label_file)  # 加载验证码的标签
        self.transform = transform

    def __getitem__(self, idx):
        # 每个图片，其中idx为数据索引,0、1...499，根据索引读取验证码图像
        img_name = os.path.join(self.root_dir, '%.4d.jpg' % idx)
        image = Image.open(img_name)

        # 对应标签，形式为【4 8 2 5】
        labels = self.label[idx]

        # 对数据预处理
        if self.transform:
            image = self.transform(image)

        # 返回每个训练样本及对应的标签
        return image, labels

    # 返回数据集的大小
    def __len__(self):
        return (self.label.shape[0])


# 自定义卷积网络模型
class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=2),  # 验证码的大小为 [3, 60, 160]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2),  # [batch_size, 32, 30, 80]

            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2),  # [batch_size, 64, 15, 40]

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2)  # [batch_size, 64, 7, 20]
        )

        self.fc1 = nn.Linear(64 * 7 * 20, 512)
        self.fc2 = nn.Linear(512, 40)  # 每个图片中有4个数字，每个数字为10分类，所以为40个输出

    def forward(self, x):
        # 使用卷积提取特征
        x = self.conv(x)  # [batch_size, 64, 7, 20]

        # 将特征图拉伸
        x = x.view(x.size(0), -1)  # [batch_size, 64 * 7 * 30] 或 [batch_size, 8960]

        # 使用输出层进行分类
        x = self.fc1(x)  # [batch_size, 512]
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)  # [batch_size, 40]

        return x


# 自定义损失函数
def loss_function(output, label):
    loss = nn.CrossEntropyLoss()  # 使用CrossEntropyLoss

    # 将验证码中每个数字作为一个独立样本，然后利用交叉熵进行分类
    output = output.contiguous().view(-1, 10)  # 将其转化为 [batch_size * 4, 10]
    label = label.contiguous().view(-1)  # [batch_size * 4, ]

    total_loss = loss(output, label)  # 交叉熵

    return total_loss


# 打开日志文件
def open_log_file(file_name=None):
    if not os.path.exists("./results"):
        os.mkdir("./results")
    file = open('./results/' + file_name, 'w', encoding='utf-8')
    return file


# 关闭日志文件
def close_log_file(file=None):
    file.close()


# 打印日志信息
def log(msg='', file=None, print_msg=True, end='\n'):
    if print_msg:
        print(msg)  # 控制台打印信息

    now = datetime.datetime.now()  # 获取当前时间
    t = str(now.year) + '/' + str(now.month) + '/' + str(now.day) + ' ' \
        + str(now.hour).zfill(2) + ':' + str(now.minute).zfill(2) + ':' + str(now.second).zfill(2)

    if isinstance(msg, str):
        lines = msg.split('\n')
    else:
        lines = [msg]

    for line in lines:
        if line == lines[-1]:
            file.write('[' + t + '] ' + str(line) + end)
        else:
            file.write('[' + t + '] ' + str(line))


file_path = './data/image/'  # 验证码图片目录
label_path = 'data/label.txt'  # 标签文件
model_path = 'checkpoints/best_model.pkl'  # 模型权重文件
batch_size = 128  # 批次大小
epochs = 30  # 迭代轮数
learning_rate = 0.003  # 学习率

# 形成数据集
dataset = MyDataset(file_path, label_path, transform=transforms.ToTensor())

# 形成迭代器
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 数据集大小
dataset_size = len(dataset)

# 定义卷积网络模型
model = ConvNet()

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 最好模型的权重，以及准确率
best_model = model.state_dict()
best_acc = 0.0

# 打开日志文件
file = open_log_file(file_name='ConvNet')

for epoch in range(epochs):
    epoch_acc = 0  # 每个epoch分类准确率
    epoch_count = 0  # 每个epoch正确分类数目
    epoch_loss = 0  # 每个epoch的loss

    if epoch == 0:
        log('【模型结构】', file, False)
        log(model, file, False)

    for x, y in dataloader:
        optimizer.zero_grad()
        pred = model(x)

        loss = loss_function(pred, y.long())
        epoch_loss += loss.item()

        # 每个batch正确分类数目
        epoch_count += pred.contiguous().view(-1, 10).argmax(axis=1).eq(y.contiguous().view(-1)).sum().item()

        loss.backward()
        optimizer.step()

        epoch_acc = epoch_count / len(y) * 4
        epoch_loss /= len(y) * 4

    log("【EPOCH: 】%s" % str(epoch + 1), file, True)
    log("训练损失为：{:.4f}".format(epoch_loss) + '\t' + "训练精度为：{:.4f}".format(epoch_acc), file, True)

    # 保存最优模型参数
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model = model.state_dict()

    # 在最后一个epoch训练完毕后将模型保存
    if not os.path.exists("./checkpoints"):
        os.mkdir("./checkpoints")
    if epoch == epochs - 1:
        # 保存模型
        torch.save(best_model, model_path)

print('【Finished Training！】')

# 关闭日志文件
close_log_file(file)
