# creating correlation data from neurofinder
import config as c
import data
import corr
import network as n
import torch

torch.cuda.empty_cache()
print(torch.cuda.get_device_name())

nf_0000 = data.NeurofinderDataset('data/training_data/aneurofinder.00.00')
corr_nf_0000 = data.create_corr_data('data/training_data/neurofinder.00.00')

print(corr_nf_0000['correlations'].size())
print(corr_nf_0000['labels'].size())

data.save_numpy_to_h5py(data_array=corr_nf_0000['correlations'].numpy(), label_array=corr_nf_0000['labels'].numpy(),
                        file_name='corr_nf_0000', file_path='data/corr/small_star/full/')

corr_nf_0001 = data.create_corr_data('data/training_data/neurofinder.00.01')

print(corr_nf_0001['correlations'].size())
print(corr_nf_0001['labels'].size())

data.save_numpy_to_h5py(data_array=corr_nf_0001['correlations'].numpy(), label_array=corr_nf_0001['labels'].numpy(),
                        file_name='corr_nf_0001', file_path='data/corr/small_star/full/')

model_UNetMS = n.UNetMS()
print(model_UNetMS)

input = torch.from_numpy(data.load_numpy_from_h5py(file_name='corr_nf_0000', file_path='data/corr/small_star/full/')[0])
input = torch.unsqueeze(input, dim=0)
print(input.size())
print(input.type())
print(input[0, 0, 0, 0])
out = model_UNetMS(input.double())
print(out)
print(out.size())
