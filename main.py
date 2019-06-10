import sys
import future        # pip install future
# import builtins      # pip install future
import past          # pip install future
import six           # pip install six


for path in ['/net/hcihome/storage/mvspreng/PycharmProjects/calcium-imaging',
             '/export/home/mvspreng/PycharmProjects/calcium-imaging',
             '/export/home/mvspreng/anaconda3/envs/testpy2/lib/python27.zip',
             '/export/home/mvspreng/anaconda3/envs/testpy2/lib/python2.7',
             '/export/home/mvspreng/anaconda3/envs/testpy2/lib/python2.7/plat-linux2',
             '/export/home/mvspreng/anaconda3/envs/testpy2/lib/python2.7/lib-tk',
             '/export/home/mvspreng/anaconda3/envs/testpy2/lib/python2.7/lib-old',
             '/export/home/mvspreng/anaconda3/envs/testpy2/lib/python2.7/lib-dynload',
             '/export/home/mvspreng/.local/lib/python2.7/site-packages',
             '/export/home/mvspreng/anaconda3/envs/testpy2/lib/python2.7/site-packages']:
    if path not in sys.path:
        sys.path.append(path)

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
import torch.nn as nn

import config as c
import data
import corr
import network as n
import visualization as v
import training as t

writer = SummaryWriter(log_dir='training_log/' + str(c.tb['loss_name']) + '/')

train = c.training['train']
dtype = c.data['dtype']
if c.cuda['use_mult']:
    device = c.cuda['mult_device']
else:
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
random_sampler = torch.utils.data.RandomSampler(comb_dataset, replacement=True, num_samples=2000)
dataloader = DataLoader(comb_dataset, batch_size=20, num_workers=0, sampler=random_sampler)
print('Initialized Dataloader')

model = n.UNetMS()
if c.cuda['use_mult']:
    model = nn.DataParallel(model, device_ids=c.cuda['use_devices'])
model.to(device)
model.type(dtype)

# loading old weights
try:
    model.load_state_dict(torch.load('model/model_weights_' + str(c.tb['loss_name']) + '.pt'))
    model.eval()
    print('Loaded Model!')
except IOError:
    print('No Model to load from!')
    print('Initializing Weights and Bias')
    model.apply(t.weight_init)
    print('Finished Initializing')

if train:

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(nb_epochs):
        # optimizer = optim.Adam(model.parameters(), lr=t.poly_lr(epoch, nb_epochs, base_lr=lr, exp=0.95))
        running_loss = 0.0
        for index, batch in enumerate(dataloader):
            input = batch['image']
            label = batch['label']

            input.requires_grad = True

            '''Debugging'''
            if c.debug['add_emb']:
                writer.add_embedding(mat=input[0].detach().view(c.UNet['input_channels'], -1).t().cpu().numpy(),
                                     tag='Before ' + str(epoch) + str(index),
                                     metadata=label[0].view(-1).detach().cpu().numpy().astype(int), global_step=epoch)
            # if c.debug['umap_img'] and index % c.debug['print_img_steps'] == 0:
            #     figure = v.draw_umap(15, 0.1, 2, 'cosine', 'Input Features Projection',
            #                          data=input[0].detach().view(c.UNet['input_channels'], -1).t().cpu().numpy(),
            #                          color=label[0].view(-1).detach().cpu().numpy().astype(int))
            #     writer.add_figure(figure=figure, tag='Input Features Projection')
            #     if c.debug['print_img']:
            #         plt.show()

            # ignoring samples where neuron density too low
            # if (label.nonzero().size(0)) / (img_size ** 2) <= c.training['min_neuron_pixels']:
            #     continue

            # zero the parameter gradients
            optimizer.zero_grad()

            output = model(input)

            '''Debugging'''
            if c.debug['add_emb']:
                writer.add_embedding(
                    mat=output[0, -1, :, :, :].detach().view(c.UNet['embedding_dim'], -1).t().cpu().numpy(),
                    tag='After ' + str(epoch) + str(index),
                    metadata=label[0].view(-1).detach().cpu().numpy().astype(int), global_step=epoch)
            # if c.debug['umap_img'] and index % c.debug['print_img_steps'] == 0:
            #     figure = v.draw_umap(15, 0.1, 2, 'cosine', 'Embedding Projection after Model',
            #                          data=output[0, -1, :, :, :].detach().view(c.UNet['embedding_dim'],
            #                                                                    -1).t().cpu().numpy(),
            #                          color=label[0].view(-1).detach().cpu().numpy().astype(int))
            #     writer.add_figure(figure=figure, tag='Predicted Embeddings')
            #     if c.debug['print_img']:
            #         plt.show()

            loss = n.get_batch_embedding_loss(embedding_list=output, labels_list=label, device=device, dtype=dtype)

            writer.add_scalar('Training Loss', loss.detach().cpu().numpy())

            loss.backward()

            # for param in model.parameters():
            #     print(param.grad.data.sum())
            #     # print(param)

            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (epoch * dataloader.__len__() + index) % 1 == 0:  # print every mini-batch
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, index + 1, running_loss / 1))
                running_loss = 0.0


        print('Saved Model After Epoch')
        torch.save(model.state_dict(), 'model/model_weights_' + str(c.tb['loss_name']) + '.pt')

    writer.close()

    print('Saved Model')
    torch.save(model.state_dict(), 'model/model_weights_' + str(c.tb['loss_name']) + '.pt')

    print('Finished Training')

if not train:
    random_sampler = torch.utils.data.RandomSampler(comb_dataset, replacement=True, num_samples=20000)
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
