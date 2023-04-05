import random
from captcha.image import ImageCaptcha
import os

# 数字
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# 小写字母
lower_letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']

# 大写字母
upper_letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z']

# 验证码图片保存目录
data_path = 'data'


# 获取验证码文本串
def random_captcha_text(char_set=number, captcha_size=4):  # 可以设置只用来生成数字
    captcha_text = []  # 验证码文本，默认4位

    for i in range(captcha_size):
        c = random.choice(char_set)  # 随机获取一个字符
        captcha_text.append(c)

    return captcha_text


# 随机产生验证码图片
def gen_captcha_text_and_image(m):
    image = ImageCaptcha()
    captcha_text = random_captcha_text()  # 生成验证码文本串，默认4位
    captcha_text = ' '.join(captcha_text)  # 生成标签 [4 3 5 8]

    if not os.path.exists("./data/image"):
        os.mkdir("./data/image")

    # 保存验证码图片
    image.write(captcha_text, "./data/image/" + '%.4d' % m + '.jpg')  # 保存图片

    # 将标签信息写入
    with open(data_path + "/label.txt", "a") as f:
        f.write(captcha_text)
        f.writelines("\n")


# 随机产生500个验证码图片
for m in range(1000):
    gen_captcha_text_and_image(m)
