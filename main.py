# creating correlation data from neurofinder
import config as c
import data
import corr
import network as n
import torch

dtype = c.data['dtype']
device = c.cuda['device']


torch.cuda.empty_cache()

# print(torch.cuda.get_device_name())
#
# nf_0000 = data.NeurofinderDataset('data/training_data/neurofinder.00.00')
# corr_nf_0000 = data.create_corr_data('data/training_data/neurofinder.00.00')
#
# print(corr_nf_0000['correlations'].size())
# print(corr_nf_0000['labels'].size())
#
# data.save_numpy_to_h5py(data_array=corr_nf_0000['correlations'].numpy(), label_array=corr_nf_0000['labels'].numpy(),
#                         file_name='corr_nf_0000', file_path='data/corr/small_star/full/')
#
# corr_nf_0001 = data.create_corr_data('data/training_data/neurofinder.00.01')
#
# print(corr_nf_0001['correlations'].size())
# print(corr_nf_0001['labels'].size())
#
# data.save_numpy_to_h5py(data_array=corr_nf_0001['correlations'].numpy(), label_array=corr_nf_0001['labels'].numpy(),
#                         file_name='corr_nf_0001', file_path='data/corr/small_star/full/')


# input_test = torch.tensor(
#     data.load_numpy_from_h5py(
#         file_name='corr_nf_0000', file_path='data/corr/small_star/full/')[0], dtype=dtype, device=device)
#
# labels = torch.tensor(
#     data.load_numpy_from_h5py(
#         file_name='corr_nf_0000', file_path='data/corr/small_star/full/')[1], dtype=dtype, device=device)
#
# print(input_test.size(), labels.size())
#
# input_test = torch.unsqueeze(input_test, dim=0)


input_test = torch.rand(1, 10, 32, 32, dtype=dtype, device=device, requires_grad=True)
labels = torch.randint(0, 10, (32, 32), dtype=dtype, device=device)

model = n.UNetMS()
model = model.type(dtype)
model = model.to(device)

print(model.parameters().__sizeof__())

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

for t in range(10):
    # Forward pass: Compute predicted y by passing x to the model
    embedding_list = model(input_test)

    # Compute and print loss
    loss = n.get_embedding_loss(embedding_list, labels, device=device, dtype=dtype)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# model = n.UNet()
# model = model.type(dtype)
# model = model.to(device)
#
# print(model.parameters().__sizeof__())
#
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
#
# for t in range(10):
#     # Forward pass: Compute predicted y by passing x to the model
#     embedding_list = model(input_test)
#
#     print(embedding_list.size())
#
#     # Compute and print loss
#     loss = n.embedding_loss(embedding_list, labels, device=device, dtype=dtype)
#     print(t, loss.item())
#
#     # Zero gradients, perform a backward pass, and update the weights.
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# model = n.MS()
# model = model.type(dtype)
# model = model.to(device)
#
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
#
# for t in range(10):
#     # Forward pass: Compute predicted y by passing x to the model
#     embedding_list = model(input_test)
#
#     print(embedding_list.size())
#
#     # Compute and print loss
#     loss = n.embedding_loss(embedding_list, labels, device=device, dtype=dtype)
#     print(t, loss.item())
#
#     # Zero gradients, perform a backward pass, and update the weights.
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()