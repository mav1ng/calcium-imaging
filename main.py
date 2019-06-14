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

# print(sys.path)

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

from torchsummary import summary

writer = SummaryWriter(log_dir='training_log/' + str(c.tb['loss_name']) + '/')

train = c.training['train']
dtype = c.data['dtype']
batch_size = c.training['batch_size']

if c.cuda['use_mult']:
    device = c.cuda['mult_device']
else:
    device = c.cuda['device']

lr = c.training['lr']
nb_epochs = c.training['nb_epochs']
img_size = c.training['img_size']

torch.cuda.empty_cache()


transform = transforms.Compose([data.CorrRandomCrop(img_size, summary_included=True)])
comb_dataset = data.CombinedDataset(corr_path='data/corr/starmy/sliced/slice_size_100/', sum_folder='data/sum_img/',
                                    transform=None, device=device, dtype=dtype)


print('Loaded the Dataset')
random_sampler = torch.utils.data.RandomSampler(comb_dataset, replacement=True, num_samples=2000)
dataloader = DataLoader(comb_dataset, batch_size=batch_size, num_workers=0, sampler=random_sampler)
print('Initialized Dataloader')

model = n.UNetMS()
if c.cuda['use_mult']:
    model = nn.DataParallel(model, device_ids=c.cuda['use_devices']).cuda()
else:
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
    criterion = n.EmbeddingLoss().cuda()
    if c.cuda['use_mult']:
        criterion = nn.DataParallel(criterion, device_ids=c.cuda['use_devices']).cuda()

    for epoch in range(nb_epochs):
        # optimizer = optim.Adam(model.parameters(), lr=t.poly_lr(epoch, nb_epochs, base_lr=lr, exp=0.95))
        running_loss = 0.0
        for index, batch in enumerate(dataloader):
            input = batch['image'].cuda()
            label = batch['label'].cuda()

            input = comb_dataset[0]['image'][:, :64, :64].view(1, 28, 64, 64).cuda()
            label = comb_dataset[0]['label'][:64, :64].view(1, 64, 64).cuda()

            input.requires_grad = True
            label.requires_grad = True

            torch.autograd.set_detect_anomaly(True)

            '''Debugging'''
            if c.debug['add_emb']:
                writer.add_embedding(mat=input[0].detach().view(c.UNet['input_channels'], -1).t().cpu().numpy(),
                                     tag='Before ' + str(epoch) + str(index),
                                     metadata=label[0].view(-1).detach().cpu().numpy().astype(int), global_step=epoch)

            output = model(input)

            '''Debugging'''
            if c.debug['add_emb']:
                writer.add_embedding(
                    mat=output[0, -1, :, :, :].detach().view(c.UNet['embedding_dim'], -1).t().cpu().numpy(),
                    tag='After ' + str(epoch) + str(index),
                    metadata=label[0].view(-1).detach().cpu().numpy().astype(int), global_step=epoch)



            loss = criterion(output, label)
            if c.cuda['use_mult']:
                loss = n.scaling_loss(loss, batch_size, c.cuda['use_devices'].__len__())
                writer.add_scalar('Training Loss', loss.detach())
            else:
                writer.add_scalar('Training Loss', loss.item())

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # for param in model.parameters():
            #     print(param.grad.data.sum())
            #     # print(param)


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
    random_sampler = torch.utils.data.RandomSampler(comb_dataset, replacement=True, num_samples=20)
    dataloader = DataLoader(comb_dataset, batch_size=1, shuffle=True, num_workers=0)
    print('Start Testing')
    for index, batch in enumerate(dataloader):
        input = batch['image']
        label = batch['label']

        '''Debugging'''
        if c.debug['add_emb']:
            writer.add_embedding(mat=input[0].detach().view(c.UNet['input_channels'], -1).t().cpu().numpy(),
                                 tag='Before ' + str(index),
                                 metadata=label[0].view(-1).detach().cpu().numpy().astype(int), global_step=0)

        output = model(input)

        if c.debug['add_emb']:
            writer.add_embedding(
                mat=output[0, -1, :, :, :].detach().view(c.UNet['embedding_dim'], -1).t().cpu().numpy(),
                tag='After ' + str(index),
                metadata=label[0].view(-1).detach().cpu().numpy().astype(int), global_step=0)

        print('Loss: ' + str(
            n.get_batch_embedding_loss(embedding_list=output, labels_list=label, device=device, dtype=dtype)))

    print('Finished Testing')



'''_______________________________________________________________________________________________________'''
#
# import time
# import random
# import numpy as np
# import neurofinder as nf
#
# def main():
#
#     device_ids = c.cuda['use_devices']
#     lr = c.training['lr']
#     resume = c.training['resume']
#     start_epoch = 0
#     img_size = c.training['img_size']
#     num_workers = c.data['num_workers']
#     num_samples = c.data['num_samples']
#     nb_epochs = c.training['nb_epochs']
#     val_freq = c.val['val_freq']
#
#
#     if c.cuda['use_mult']:
#         device = c.cuda['mult_device']
#     else:
#         device = c.cuda['device']
#
#         model = n.UNetMS()
#
#
#     if c.cuda['use_mult']:
#         model = nn.DataParallel(model, device_ids=c.cuda['use_devices']).cuda()
#     else:
#         model.to(device)
#     model.type(dtype)
#
#
#     # define loss function (criterion) and optimizer
#     # perhaps add additional loss functions if this one does not work
#     emb_loss = n.EmbeddingLoss().cuda()
#
#     if c.cuda['use_mult']:
#         emb_loss = nn.DataParallel(emb_loss, device_ids=c.cuda['use_devices']).cuda()
#
#     criterion = [emb_loss]
#
#     # # creating different parameter groups
#     base_params = list(map(id, model.parameters()))
#
#     # optimizer - with lr initialized accordingly
#     optimizer = torch.optim.Adam({'params': model.parameters(), 'lr': lr})
#     optimizer = optimizer.add_param_group(model.parameters())
#
#     if resume:
#         if os.path.isfile(resume):
#             print("=> loading checkpoint '{}'".format(resume))
#             checkpoint = torch.load(resume)
#             start_epoch = checkpoint['epoch']
#             best_f1 = checkpoint['best_f1']
#             model.load_state_dict(checkpoint['state_dict'])
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             print("=> loaded checkpoint '{}' (epoch {})"
#                   .format(resume, checkpoint['epoch']))
#         else:
#             print("=> no checkpoint found at '{}'".format(resume))
#             best_f1 = 0.
#     else:
#         best_f1 = 0.
#
#     # models are save only when their loss obtains the best value in the validation
#     f1track = 0
#
#     print 'There are %d parameter groups' % len(optimizer.param_group)
#     print 'Initial base params lr: %f' % optimizer.param_group[0]['lr']
#
#     # data preparation, loaders
#     # cudnn.benchmark = True
#
#     # preparing the training loader
#     transform_train = transforms.Compose([data.CorrRandomCrop(img_size, summary_included=True)])
#     random_sampler = torch.utils.data.RandomSampler(comb_dataset, replacement=True, num_samples=num_samples)
#
#     train_loader = torch.utils.data.DataLoader(
#         data.CombinedDataset(corr_path='data/corr/starmy/sliced/slice_size_100/', sum_folder='data/sum_img/',
#                              transform=transform_train, device=device, dtype=dtype), batch_size=batch_size,
#         sampler=random_sampler, num_workers=num_workers, pin_memory=True)
#     print 'Training loader prepared.'
#
#     # preparing validation loader
#     transform_val = transforms.Compose([data.CorrCorrect(summary_included=True)])
#     val_loader = torch.utils.data.DataLoader(
#         data.CombinedDataset(corr_path='data/corr/starmy/sliced/slice_size_100/', sum_folder='data/sum_img/',
#                              transform=transform_val, device=device, dtype=dtype, test=True), batch_size=batch_size,
#         shuffle=True, num_workers=num_workers, pin_memory=True)
#     print 'Validation loader prepared.'
#
#     # run epochs
#     for epoch in range(start_epoch, nb_epochs):
#
#         # train for one epoch
#         train(train_loader, model, criterion, optimizer, epoch)
#
#         # evaluate on validation set
#         if (epoch + 1) % val_freq == 0 and epoch != 0:
#             'rewrite THIS PART BECAUSE USING NEUROFINDER METRIC'
#             f1_metric = validate(val_loader, model)
#
#             # check patience
#             if f1_metric <= best_f1:
#                 f1track += 1
#             else:
#                 f1track = 0
#
#             # save the best model
#             is_best = f1_metric < best_f1
#             best_f1 = min(f1_metric, best_f1)
#             save_checkpoint({
#                 'epoch': epoch + 1,
#                 'state_dict': model.state_dict(),
#                 'best_val': best_f1,
#                 'optimizer': optimizer.state_dict(),
#                 'valtrack': f1track,
#                 'curr_val': f1_metric,
#             }, is_best)
#
#             print '** Validation: %f (best) - %d (f1track)' % (best_f1, f1track)
#
#
# def train(train_loader, model, criterion, optimizer, epoch):
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     cos_losses = AverageMeter()
#
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#
#     # switch to train mode
#     model.train()
#
#     end = time.time()
#     for i, (input, target) in enumerate(train_loader):
#
#         # measure data loading time
#         data_time.update(time.time() - end)
#
#         input_var = list()
#         for j in range(len(input)):
#             input_var.append(torch.autograd.Variable(input[j]).cuda())
#
#         target_var = list()
#         for j in range(len(target)):
#             target[j] = target[j].cuda(async=True)
#             target_var.append(torch.autograd.Variable(target[j]))
#
#         # compute output
#         output = model(input)
#
#         loss = criterion(output, target)
#         # measure performance and record loss
#         cos_losses.update(loss.item(), input.size())
#
#         # compute gradient and do Adam step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         print('Epoch: {0}\t'
#               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#               'Param ({param}))\t'.format(
#             epoch, loss=cos_losses, param=optimizer.param_groups[0]['lr']))
#
#
# def validate(val_loader, model):
#
#     nf_threshold = c.val['nf_threshold']
#
#     # switch to evaluate mode
#     model.eval()
#
#     end = time.time()
#
#     f1_metric = 0.
#     (recall, precision) = (0., 0.)
#     count = 0
#
#     for i, (input, target) in enumerate(val_loader):
#         input_var = list()
#         for j in range(len(input)):
#             input_var.append(torch.autograd.Variable(input[j], volatile=True).cuda())
#         target_var = list()
#         for j in range(len(target)):
#             target[j] = target[j].cuda()
#             target_var.append(torch.autograd.Variable(target[j], volatile=True))
#
#         # compute output
#         output = model(input)
#
#         '''Write Clustering METHOD such that Embeddings get clustered
#         Write From Labels to Coordinates Method
#         Write Method that saves input and target data as Jason file appropriately'''
#
#         predict = cluster(output)
#
#         f1_metric_ = 0.
#         (recall_, precision_) = (0., 0.)
#
#         for i in range(input.size(0)):
#             (mask_true, mask_predict) = (data.toCoords(label[i]), data.toCoords(predict[i]))
#             data.write_to_json(mask_true, 'data/val_mask/mask_true.json')
#             data.write_to_json(mask_predict, 'data/val_mask/mask_predict.json')
#             true_mask = nf.load('data/val_mask/mask_true.json')
#             predict_mask = nf.load('data/val_mask/mask_predict.json')
#             (recall_, precision_) = (recall_, precision_) + nf.centers(true_mask, predict_mask, threshold=nf_threshold)
#             f1_metric_ = f1_metric_ + 2 * (recall_ * precision_) / (recall_ + precision_)
#
#         f1_metric_ = f1_metric_ / input.size(0)
#         recall_ = recall_ / input.size(0)
#         precision_ = precision_ / input.size()
#
#         f1_metric = f1_metric + f1_metric_
#         recall = recall + recall_
#         precision = precision + precision_
#
#         count = i
#
#     f1_metric = f1_metric / (count + 1)
#     recall = recall / (count + 1)
#     precision = precision / (count + 1)
#
#     print('* F1 Metric {f1:.4f}\t'
#         'Recall {recall}\t'
#         'Precision'.format(f1=f1_metric, recall=recall, precision=precision))
#
#     return f1_metric
#
#
# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     snapshots = c.data['snapshots']
#     filename = snapshots + 'model_e%03d_v-%.3f.pth.tar' % (state['epoch'], state['best_val'])
#     if is_best:
#         torch.save(state, filename)
#
#
# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
#
#
# if __name__ == '__main__':
#     main()