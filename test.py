import sys
for i in ['/net/hcihome/storage/mvspreng/PycharmProjects/calcium-imaging',
          '/export/home/mvspreng/anaconda3/envs/testpy2/lib/python27.zip',
          '/export/home/mvspreng/anaconda3/envs/testpy2/lib/python2.7',
          '/export/home/mvspreng/anaconda3/envs/testpy2/lib/python2.7/plat-linux2',
          '/export/home/mvspreng/anaconda3/envs/testpy2/lib/python2.7/lib-tk',
          '/export/home/mvspreng/anaconda3/envs/testpy2/lib/python2.7/lib-old',
          '/export/home/mvspreng/anaconda3/envs/testpy2/lib/python2.7/lib-dynload',
          '/export/home/mvspreng/.local/lib/python2.7/site-packages',
          '/export/home/mvspreng/anaconda3/envs/testpy2/lib/python2.7/site-packages']:
    if i not in sys.path:
        sys.path.append(i)
import numpy as np
import torch
import torch.nn.functional as F
import config as c
import network as n
import json
import neurofinder as nf



with torch.no_grad():
    self.bs = x_in.size(0)
    self.emb = x_in.size(1)
    self.w = x_in.size(2)
    self.h = x_in.size(3)


x = x_in.view(self.bs, self.emb, -1)

y = torch.zeros(self.nb_iterations + 1, self.emb, self.w * self.h)
out = torch.zeros(self.bs, self.nb_iterations + 1, self.emb, self.w, self.h,
                  device=self.device, dtype=self.dtype)

# iterating over all samples in the batch
for b in range(self.bs):
    y[0, :, :] = x[b, :, :]
    for t in range(1, self.nb_iterations):
        y[t, :, :] = y[t - 1, :, :]
        kernel_mat = torch.exp(torch.mul(self.kernel_bandwidth, mm(y[t, :, :].t(), y[t, :, :])))
        diag_mat = torch.diag(
            mm(kernel_mat.t(), torch.ones((self.w * self.h, 1), device=self.device, dtype=self.dtype)).squeeze(dim=1),
            diagonal=0)

        y[t, :, :] = mm(y[t, :, :], torch.add(torch.mul(self.step_size, mm(kernel_mat, torch.inverse(diag_mat))),
                                              torch.mul(1. - self.step_size, torch.eye(self.w * self.h))))
    out[b, :, :, :, :] = y.view(self.nb_iterations, self.emb, self.w, self.h)

return out
