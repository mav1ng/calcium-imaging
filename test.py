import numpy as np
import torch
import torch.nn.functional as F
import config as c
import network as n
import json
import neurofinder as nf

a = torch.rand(10, 5, 3, 3, device=torch.device('cuda'))
l = torch.randint(0, 10, (10, 10), device=torch.device('cuda'))
b = torch.randint(0, 10, (10, 10), device=torch.device('cuda'))

print(l)

def toCoords(mask):
    """
    Method that returns the Coordinates of the Regions in the Mask
    :param mask:
    :return:
    """
    unique = torch.unique(mask)
    coords = []
    for _, label in enumerate(unique):
        coords.append({'coordinates': (mask == label).nonzero().cpu().numpy().tolist()})
    return coords


data = toCoords(l)
data2 = toCoords(b)

with open('data/val_mask/data.json', 'w') as outfile:
    json.dump(data, outfile)

with open('data/val_mask/data2.json', 'w') as outfile:
    json.dump(data2, outfile)

with open('data/val_mask/data.json', 'r') as json_file:
    data1 = json.load(json_file)

with open('data/val_mask/data2.json', 'r') as json_file:
    data2 = json.load(json_file)

# print(data)
# print(data2)

a = nf.load('data/val_mask/data.json')
b = nf.load('data/val_mask/data2.json')

# print(a)
# print(b)

print(nf.match(a, b, threshold=np.inf))
print(nf.centers(a, b, threshold=0.2))