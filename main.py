import sys
import future        # pip install future
import builtins      # pip install future
import past          # pip install future
import six           # pip install six


if '/export/home/mvspreng/PycharmProjects' not in sys.path:
    sys.path.append('/export/home/mvspreng/PycharmProjects')

if '/export/home/mvspreng/PycharmProjects/calcium-imaging' not in sys.path:
    sys.path.append('/export/home/mvspreng/PycharmProjects/calcium-imaging')

if '/net/hcihome/storage/mvspreng/PycharmProjects/calcium-imaging' not in sys.path:
    sys.path.append('/net/hcihome/storage/mvspreng/PycharmProjects/calcium-imaging')

if '/export/home/mvspreng/anaconda3/lib/python37.zip' not in sys.path:
    sys.path.append('/export/home/mvspreng/anaconda3/lib/python37.zip')

if '/export/home/mvspreng/anaconda3/lib/python3.7' not in sys.path:
    sys.path.append('/export/home/mvspreng/anaconda3/lib/python3.7')

if '/export/home/mvspreng/anaconda3/lib/python3.7/lib-dynload' not in sys.path:
    sys.path.append('/export/home/mvspreng/anaconda3/lib/python3.7/lib-dynload')

if '/export/home/mvspreng/anaconda3/lib/python3.7/site-packages' not in sys.path:
    sys.path.append('/export/home/mvspreng/anaconda3/lib/python3.7/site-packages')

print(sys.path)

from torch import optim
import os
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.tensorboard import SummaryWriter
import umap

import config as c
import data
import corr
import network as n
import visualization as v
import training as t

writer = SummaryWriter(log_dir='training_log/')

train = c.training['train']
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
    model.load_state_dict(torch.load('model/model_weights_iter_0.pt'))
    model.eval()
    print('Loaded Model!')
except FileNotFoundError:
    print('No Model to load from!')
    print('Initializing Weights and Bias')
    model.apply(t.weight_init)
    print('Finished Initializing')

if train:

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(nb_epochs):
        # optimizer = optim.Adam(model.parameters(), lr=t.poly_lr(epoch, nb_epochs, base_lr=lr, exp=0.95))
        running_loss = 0.0
        for i in range(100):
            for index, batch in enumerate(dataloader):
                input = batch['image']
                label = batch['label']

                '''Debugging'''
                if c.debug['umap_img'] and i % c.debug['print_img_steps'] == 0:
                    figure = v.draw_umap(15, 0.1, 2, 'cosine', 'Input Features Projection',
                                         data=input[0].detach().view(c.UNet['input_channels'], -1).t().cpu().numpy(),
                                         color=label[0].view(-1).detach().cpu().numpy().astype(int))
                    writer.add_figure(figure=figure, tag='Input Features Projection')
                    if c.debug['print_img']:
                        plt.show()

                # ignoring samples where neuron density too low
                # if (label.nonzero().size(0)) / (img_size ** 2) <= c.training['min_neuron_pixels']:
                #     continue

                # zero the parameter gradients
                optimizer.zero_grad()

                output = model(input)

                '''Debugging'''
                if c.debug['umap_img'] and i % c.debug['print_img_steps'] == 0:
                    figure = v.draw_umap(15, 0.1, 2, 'cosine', 'Embedding Projection after Model',
                                         data=output[0, -1, :, :, :].detach().view(c.UNet['embedding_dim'],
                                                                                    -1).t().cpu().numpy(),
                                         color=label[0].view(-1).detach().cpu().numpy().astype(int))
                    writer.add_figure(figure=figure, tag='Predicted Embeddings')
                    if c.debug['print_img']:
                        plt.show()

                loss = n.get_batch_embedding_loss(embedding_list=output, labels_list=label, device=device, dtype=dtype)

                writer.add_scalar('Training Loss', loss.detach().cpu().numpy())

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if (epoch * dataloader.__len__() + index) % 1 == 0:  # print every mini-batch
                    print('[%d, %5d] loss: %.5f' %
                          (epoch + 1, i + 1, running_loss / 1))
                    running_loss = 0.0


        print('Saved Model After Epoch')
        torch.save(model.state_dict(), 'model/model_weights_iter_0.pt')

    writer.close()

    print('Saved Model')
    torch.save(model.state_dict(), 'model/model_weights_iter_0.pt')

    print('Finished Training')

if not train:
    dataloader = DataLoader(comb_dataset, batch_size=1, shuffle=True, num_workers=0)
    print('Start Testing')
    for index, batch in enumerate(dataloader):
        input = batch['image']
        label = batch['label']

        if (label.nonzero().size(0)) / (img_size ** 2) <= c.training['min_neuron_pixels']:
            continue


        v.draw_umap(15, 0.1, 2, 'cosine', 'Input Features Projection',
                    data=input[0].detach().view(input.size(1), -1).t().cpu().numpy(),
                    color=label[0].view(-1).detach().cpu().numpy().astype(int))
        plt.show()
        output = model(input)
        print('Loss: ' + str(
            n.get_batch_embedding_loss(embedding_list=output, labels_list=label, device=device, dtype=dtype)))
        v.draw_umap(15, 0.1, 2, 'cosine', 'Embedding Projection after Model',
                    data=output[0, -1, :, :, :].detach().view(c.UNet['embedding_dim'],
                                                              -1).t().unique(dim=0).cpu().numpy(),
                    color=label[0].view(-1).detach().cpu().numpy().astype(int))
        # v.draw_umap(15, 0.1, 2, 'cosine', 'Embedding Projection after Model',
        #             data=output[0, -1, :, :, :].detach().view(c.UNet['embedding_dim'],
        #                                                       -1).t().cpu().numpy(),
        #             color=label[0].view(-1).detach().cpu().numpy().astype(int))
        plt.show()
    print('Finished Testing')
