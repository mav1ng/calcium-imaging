# creating correlation data from neurofinder
import config as c
import data
import corr
import network as n
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

dtype = c.data['dtype']
device = c.cuda['device']

torch.cuda.empty_cache()

transform = transforms.Compose([data.CorrRandomCrop(64)])
corr_dataset = data.CorrelationDataset(folder_path='data/corr/small_star/full/', transform=transform, dtype=dtype)

dataloader = DataLoader(corr_dataset, batch_size=2, shuffle=True, num_workers=0)

model = n.UNetMS()
model.to(device)
model.type(dtype)


for index, batch in enumerate(dataloader):
    input = batch['image']
    label = batch['label']
    output = model(input)
    print(index, batch['image'].size(), batch['label'].size())

