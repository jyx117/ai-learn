import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

img_path = r'./data/image/0000.jpg'  # 测试验证码图片

img = Image.open(img_path)  # 打开图片
img = transforms.ToTensor()(img)  # 将其转换成tensor
img = torch.unsqueeze(img, dim=0)  # 处理成模型输入格式 [batch_size, 3, 60, 160]


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


# 加载模型权重
model = ConvNet()
model.load_state_dict(torch.load('./checkpoints/best_model.pkl'))

# 验证码识别
pred = model(img)
predict_captcha = pred.contiguous().view(-1, 10).argmax(axis=1).numpy().tolist()
print('验证码: ', predict_captcha)
