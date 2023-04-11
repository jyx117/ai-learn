# -*- coding = utf-8 -*-
"""
@project: ai-learn
@Author：michael
@file： test.py
@date：2023/4/11 10:59 上午
"""
import torch

X = torch.tensor([[4, 4], [4, 4], [4, 4]])
X_exp = torch.exp(X)
print('X_exp:', X, X_exp)
partition = X_exp.sum(1, keepdim=True)
print('partition:', partition)
value = X_exp / partition
print('value:', value)
