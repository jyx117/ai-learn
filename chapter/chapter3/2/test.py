# -*- coding = utf-8 -*-
"""
@project: ai-learn
@Author：michael
@file： test.py
@date：2023/4/11 9:56 上午
"""

import torch

true_w = torch.tensor([2., 3.])
true_b = 5.
w = true_w
b = true_b
X = torch.normal(0, 1, (5, len(w)))
y = torch.matmul(X, w) + b
print('y1:', y.shape, X, b, y)
y += torch.normal(0, 0.01, y.shape)
print('y2:', y.shape, y, y.reshape((-1, 1)))
