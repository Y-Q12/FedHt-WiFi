import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from torch.autograd import Variable
import pickle
# import bill
# from data import Get_dataloader
# from data import Get_dataloader
from data import Get_dataloader

from CNNDEMO import  shot
import CNNDEMO


parser = argparse.ArgumentParser(description='SHOT')
parser.add_argument('--gpu_id', type=str, nargs='?', default='2', help="device id to run")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--seed', type=int, default=400, help="random seed")#217 218 228 #234zyydata  219 220 221 223,224 235yqdata
parser.add_argument('--source_epoch', type=int, default=100, help="sourece epoch")
parser.add_argument('--traget_epoch', type=int, default=20, help="traget epoch")
parser.add_argument('--cls_par', type=float, default=0)
parser.add_argument('--ent_par', type=float, default=1) #1.0
parser.add_argument('--gent', type=bool, default=True)
parser.add_argument('--ent', type=bool, default=True)
parser.add_argument('--smooth', type=float, default=0.1)
parser.add_argument('--output', type=str, default='')
parser.add_argument('--issave', type=bool, default=True)
# parser.add_argument('--issave', type=bool, default=True)
args = parser.parse_args()
args.class_num = 7

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
SEED = args.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
# torch.backends.cudnn.deterministic = True

globalF_parameters= {}

# global_parameters = {}  #加载全局模型
netF = CNNDEMO.CNN().cuda()

for key,var in netF.state_dict().items():
    globalF_parameters[key] = var.clone()
    # print(var.shape)

source_domain = ['A','B']
df =len(source_domain)
for k in range(10):
    print("communicate round {}".format(k + 1))
    sum_parametersF = None

    for i in source_domain:
        print('source_domain{}'.format(i))
        args.output_dir = osp.join(args.output, 'seed' + str(args.seed) + '{}'.format(i))
        Shot_1 = shot(args, source=i, target='C',globalF_parameters =globalF_parameters,k =k)
        # args.output_dir = osp.join(args.output, 'seed' + str(args.seed) + '{}'.format(i))

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        # if not osp.exists(osp.join(args.output_dir + '/source_F.pt')):  #socure shi fou chongxin xunlian
        # Shot_1.train_source()
        # args.out_file = open(osp.join(args.output_dir, 'log_src.txt'), 'w')
        # args.out_file.write(Shot_1.print_args() + '\n')
        # args.out_file.flush()
        #
        # args.savename = 'par_' + str(args.cls_par)

        net = Shot_1.train() #更新网络模型
        # print(net[0].items)
        # print('netF[0].state_dict()', net[0])
        # local_parameters = net[0].state_dict
        # print(local_parameters.items())
        # print('netB[1].state_dict()', net[1].state_dict())
        # print('netC[2].state_dict()', net[2].state_dict())

        if sum_parametersF is None:
            sum_parametersF = {}
            for key, var in net.items():
                # print('aaaaaaa',net[0].items())
                sum_parametersF[key] = var.clone()
        else:
            for var in sum_parametersF:
                sum_parametersF[var] = sum_parametersF[var] + net[var]


    for var in globalF_parameters:
        globalF_parameters[var] = (sum_parametersF[var] / df )
        # print('sum_parameters[var]',sum_parameters[var].shape)
        # print('globalF_parameters[var]',globalF_parameters[var])

    with torch.no_grad():
        # print('globalF_parameters', globalF_parameters)
        netF.load_state_dict(globalF_parameters, strict=True)
        correct = 0
        total = 0
        test_loader, train_loader = Get_dataloader()

        for i, q in enumerate(test_loader):
            test_data = Variable(q['C'].type(torch.FloatTensor)).cuda()
            test_label = q['label_C'][:, 0].cuda()
            # print(test_label)
            # net = netC(netB(netF()))
            out = netF(test_data)
            _, pre = torch.max(out.data, 1)
            correct += (pre == test_label.long()).sum().item()
            total += test_label.size(0)
            # print('total',total)
        acc = correct / total
        print('final target accuracy on the test is:', acc)
