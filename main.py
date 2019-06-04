# creating correlation data from neurofinder
from torch import optim

import config as c
import data
import corr
import network as n
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plot as p

dtype = c.data['dtype']
device = c.cuda['device']

torch.cuda.empty_cache()

transform = transforms.Compose([data.CorrRandomCrop(32)])
corr_dataset = data.CorrelationDataset(folder_path='data/corr/small_star/full/', transform=transform, dtype=dtype)

dataloader = DataLoader(corr_dataset, batch_size=4, shuffle=True, num_workers=0)

model = n.UNetMS()
model.to(device)
model.type(dtype)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for index, batch in enumerate(dataloader):
        input = batch['image']
        label = batch['label']

        # zero the parameter gradients
        optimizer.zero_grad()

        output = model(input)

        # p.plot3Dembeddings(output.clone().detach())

        loss = n.get_batch_embedding_loss(embedding_list=output, labels_list=label, device=device, dtype=dtype)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if index % 1 == 0:  # print every mini-batch
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, index + 1, running_loss))
            running_loss = 0.0



print('Finished Training')



