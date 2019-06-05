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

import plot as p
import training as t


dtype = torch.double
device = torch.device('cpu')

'''Creating Correlaations'''
'''Creating Correlation Dataset'''

for index, folder in enumerate(sorted(os.listdir('data/training_data'))):
    print(folder)
    corr = data.create_corr_data(neurofinder_path='data/training_data/' + str(folder), slicing=False, corr_form='big_star')
    data.save_numpy_to_h5py(data_array=corr['correlations'].numpy(), label_array=corr['labels'].numpy(),
                                file_name='corr_nf_' + str(index), file_path='data/corr/big_star/full/')

for index, folder in enumerate(sorted(os.listdir('data/training_data'))):
    print(folder)
    corr = data.create_corr_data(neurofinder_path='data/training_data/' + str(folder), slicing=True, slice_size=100, corr_form='big_star')
    data.save_numpy_to_h5py(data_array=corr['correlations'].numpy(), label_array=corr['labels'].numpy(),
                                file_name='corr_nf_' + str(index), file_path='data/corr/big_star/sliced/slice_size_100/')

for index, folder in enumerate(sorted(os.listdir('data/training_data'))):
    print(folder)
    corr = data.create_corr_data(neurofinder_path='data/training_data/' + str(folder), corr_form='small_star', slicing=True, slice_size=100)
    data.save_numpy_to_h5py(data_array=corr['correlations'].numpy(), label_array=corr['labels'].numpy(),
                                file_name='corr_nf_' + str(index), file_path='data/corr/small_star/sliced/slice_size_100/')


dtype = c.data['dtype']
device = c.cuda['device']
lr = c.training['lr']
nb_epochs = c.training['nb_epochs']
img_size = c.training['img_size']

torch.cuda.empty_cache()

transform = transforms.Compose([data.CorrRandomCrop(img_size)])
corr_dataset = data.CorrelationDataset(folder_path='data/corr/big_star/full/', transform=transform, dtype=dtype)

dataloader = DataLoader(corr_dataset, batch_size=1, shuffle=True, num_workers=0)

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

optimizer = optim.SGD(model.parameters(), lr=lr)

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
    optimizer = optim.SGD(model.parameters(), lr=t.poly_lr(epoch, nb_epochs, base_lr=lr, exp=0.95))
    running_loss = 0.0
    for index, batch in enumerate(dataloader):
        input = batch['image']
        label = batch['label']

        # ignoring samples where neuron density too low
        if ((label != 0).nonzero().size(0)) / (img_size ** 2) <= c.training['min_neuron_pixels']:
            pass

        # zero the parameter gradients
        optimizer.zero_grad()

        output = model(input)

        loss = n.get_batch_embedding_loss(embedding_list=output, labels_list=label, device=device, dtype=dtype)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (epoch * dataloader.__len__() + index) % 100 == 99:  # print every mini-batch
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, index + 1, running_loss / 100))
            running_loss = 0.0
            # p.plot3Dembeddings(output[:, :, :, :, :])

print('Saved Model')
torch.save(model.state_dict(), 'model/model_weights.pt')

print('Finished Training')



