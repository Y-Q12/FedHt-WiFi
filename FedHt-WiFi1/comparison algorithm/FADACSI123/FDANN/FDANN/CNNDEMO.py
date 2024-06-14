import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from torch.autograd import Variable
import torch
import pickle
# import bill
from tensorboardX import SummaryWriter
# from data import Get_dataloader
import time
from data import Get_dataloader
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.feature = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
                                      nn.BatchNorm2d(16),
                                      nn.ReLU(True),
                                      nn.MaxPool2d(kernel_size=2),
                                   nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(True),
                                   nn.MaxPool2d(kernel_size=2),

                                   nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(True),
                                   nn.MaxPool2d(kernel_size=2),

                                     )
        self.class_d = nn.Sequential(nn.Linear(48000, 256),
                                   nn.ReLU(True),
                                   nn.Linear(256, 7))


    def forward(self, x):
        # print(x.shape)
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        class_pre = self.class_d(feature)
        # print(class_pre.shape)

        return class_pre
class shot(object):
    def __init__(self,args,source,target):#,globalF_parameters,globalB_parameters,globalC_parameters
        self.args = args
        self.target = target
        # self.clients_set = {}
        self.source = source
        # print('self',self.source)
        self.lr = args.lr
        self.output = args.output
        self.output_dir = args.output_dir
        # print(self.output_dir)
        self.smooth = args.smooth
        self.class_num = args.class_num
        self.cls_par = args.cls_par
        self.gent = args.gent
        self.ent = args.ent
        self.ent_par= args.ent_par
        self.source_epoch = args.source_epoch
        self.traget_epoch = args.traget_epoch
        self.args.__dict__ = args.__dict__

        # print(self.globalF_parameters)

        # print(self.target)
        # self.train_source(args)
        # self.train_target(args)

    def test_source(self,epoch, my_net, test_loader, total_time):
        my_net.eval()

        correct = 0
        total = 0
        for i, q in enumerate(test_loader):
            test_data = Variable(q['{}'.format(self.target)].type(torch.FloatTensor)).cuda()
            test_label = q['label_{}'.format(self.target)][:, 0].cuda()
            out = my_net(test_data)

            _, pre = torch.max(out.data, 1)
            correct += (pre == test_label.long()).sum().item()
            total += test_label.size(0)
        acc = correct / total
        print('source accuracy on the test is:', acc)

        # return my_net.state_dict()
        # if acc >= 0.96:
        #     torch.save(my_net, 'ADDA_BAS.pkl')
        #     torch.save(classifier_net, 'ADDA_BACl.pkl')

        # with open('CNNBA.txt', 'a+') as f:
        #     f.write(str(acc) + str(-epoch) + '-训练至这次耗费总时间为： ' + str(total_time) + '分钟' + '\n')
        # f.close()

    def train(self):
        n_epoch = self.traget_epoch
        step_decay_weight = 0.95
        lr_decay_step = 20000
        weight_decay = 1e-6
        alpha_weight = 0.01
        beta_weight = 0.075
        momentum = 0.9

        total_time = 0
        test_loader, train_loader = Get_dataloader()
        net_S = CNN().cuda()

        optimizer_net_S = optim.Adam(net_S.parameters(), lr=0.001, betas=(0.5, 0.999))
        loss_classification = nn.CrossEntropyLoss()
        cuda = True

        if cuda:
            loss_classification = loss_classification.cuda()

        for epoch in range(n_epoch):
            start_time = time.time()
            net_S.train()

            for i, d in enumerate(train_loader):
                source_data = Variable(d['{}'.format(self.source)].type(torch.FloatTensor)).cuda()

                source_class_l = d['label_{}'.format(self.source)][:, 0].cuda()
                # train source model
                optimizer_net_S.zero_grad()

                pre = net_S(source_data)
                # print(pre.shape)
                # print(source_class_l.long().shape)

                class_loss = loss_classification(pre, source_class_l.long())
                class_loss.backward()
                optimizer_net_S.step()

                print('[%d/%d][%d/%d] loss: %.4f  ' % (
                    epoch, n_epoch, i, len(train_loader), class_loss))
            end_time = time.time()
            total_time += round((end_time - start_time) / 60, 2)
            print("训练至第{}个epoch耗费时间为： {}分钟".format(epoch, round(total_time, 2)))
            shot.test_source(self,epoch, net_S, test_loader, total_time)
        return net_S.state_dict()




