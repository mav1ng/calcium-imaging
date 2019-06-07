# creating correlation data from neurofinder
from torch import optim
import os

import config as c
import data
import corr
import network as n
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import visualization as v
import training as t
from torch.utils.tensorboard import SummaryWriter
import umap

# writer = SummaryWriter(log_dir='training_log/')

dtype = c.data['dtype']
device = c.cuda['device']
lr = c.training['lr']
nb_epochs = c.training['nb_epochs']
img_size = c.training['img_size']

torch.cuda.empty_cache()

# transform = transforms.Compose([data.CorrRandomCrop(img_size)])
# corr_dataset = data.CorrelationDataset(folder_path='data/corr/starmy/sliced/slice_size_100/', transform=transform,
#                                        dtype=dtype)
# print('Loaded the Dataset')
#
# dataloader = DataLoader(corr_dataset, batch_size=1, shuffle=True, num_workers=0)
# print('Initialized Dataloader')

transform = transforms.Compose([data.CorrRandomCrop(img_size, summary_included=True)])
comb_dataset = data.CombinedDataset(corr_path='data/corr/starmy/sliced/slice_size_100/', sum_folder='data/sum_img/',
                                    transform=transform, device=device, dtype=dtype)
print('Loaded the Dataset')
dataloader = DataLoader(comb_dataset, batch_size=20, shuffle=True, num_workers=0)
print('Initialized Dataloader')

model = n.UNetMS()
model.to(device)
model.type(dtype)

# loading old weights
try:
    model.load_state_dict(torch.load('model/model_weights.pt'))
    model.eval()
    print('Loaded Model!')
except FileNotFoundError:
    print('No Model to load from!')
    print('Initializing Weights and Bias')
    model.apply(t.weight_init)
    print('Finished Initializing')

optimizer = optim.Adam(model.parameters(), lr=lr)

# # Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
#
# # Print optimizer's state_dict
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])


for epoch in range(nb_epochs):
    # optimizer = optim.Adam(model.parameters(), lr=t.poly_lr(epoch, nb_epochs, base_lr=lr, exp=0.95))
    running_loss = 0.0
    for i in range(100):
        for index, batch in enumerate(dataloader):
            input = batch['image']
            label = batch['label']

            # v.draw_umap(15, 0.1, 2, 'cosine', 'Embedding Projection',
            #             data=input.detach().view(c.UNet['input_channels'], -1).t().cpu().numpy(),
            #             color=label.view(-1).detach().cpu().numpy().astype(int))
            # plt.show()

            # ignoring samples where neuron density too low
            if (label.nonzero().size(0)) / (img_size ** 2) <= c.training['min_neuron_pixels']:
                continue

            # zero the parameter gradients
            optimizer.zero_grad()

            output = model(input)

            loss = n.get_batch_embedding_loss(embedding_list=output, labels_list=label, device=device, dtype=dtype)

            # writer.add_scalar('Training Loss', loss)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (epoch * dataloader.__len__() + index) % 1 == 0:  # print every mini-batch
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, i + 1, running_loss / 1))
                running_loss = 0.0

                # v.draw_umap(15, 0.1, 2, 'euclidean', 'Embedding Projection after Model',
                #             data=output[-1, -1, :, :, :].detach().view(c.UNet['embedding_dim'], -1).t().cpu().numpy(),
                #             color=label.view(-1).detach().cpu().numpy().astype(int))
                # plt.show()

                # tensorboard
                # writer.add_embedding(tag='Embedding', mat=output[-1, -1, :, :, :].view(c.UNet['embedding_dim'], -1).t())


# writer.close()

print('Saved Model')
torch.save(model.state_dict(), 'model/model_weights.pt')

print('Finished Training')



