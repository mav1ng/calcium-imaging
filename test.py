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



