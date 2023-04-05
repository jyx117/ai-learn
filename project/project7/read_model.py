import torch

file = 'checkpoints/best_model.pkl'

f = open(file, 'rb')
data = torch.load(f, map_location='cpu')  # 可使用cpu或gpu
print(data)
