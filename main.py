# creating correlation data from neurofinder
import config as c
import data
import corr
import network as n
import torch
from torchvision import transforms, utils

dtype = c.data['dtype']
device = c.cuda['device']

torch.cuda.empty_cache()

transform = transforms.Compose([data.CorrRandomCrop(64)])
corr_dataset = data.CorrelationDataset(folder_path='data/corr/small_star/full/', transform=transform, dtype=dtype)

print(corr_dataset.__len__())

print(corr_dataset[0]['image'].size())