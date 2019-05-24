# creating correlation data from neurofinder
import config as c
import data
import corr
import network as n
import torch

torch.cuda.empty_cache()
print(torch.cuda.get_device_name())

nf_0000 = data.NeurofinderDataset('data/neurofinder.00.00')
corr_nf_0000 = data.create_corr_data('data/neurofinder.00.00')

print(corr_nf_0000['correlations'].size())
print(corr_nf_0000['labels'].size())

data.save_numpy_to_h5py(data_array=corr_nf_0000['correlations'].numpy(), label_array=corr_nf_0000['labels'].numpy(),
                        file_name='corr_nf_0000_full_correct', file_path='data/corr/')
