import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.randn(1, 1, 32, 32)
b = torch.randn(32, 32)
print(a)
print(b)
print(a.size())
print(a.size()[2])
for f in range(10):
    print(f)