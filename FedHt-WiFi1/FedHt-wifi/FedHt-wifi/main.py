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
from  demo import Logger
from demo import  shot
import network


parser = argparse.ArgumentParser(description='SHOT')
parser.add_argument('--gpu_id', type=str, nargs='?', default='2', help="device id to run")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--seed', type=int, default=10, help="random seed")#217 218 228 #234zyydata  219 220 221 223,224 235yqdata
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

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
SEED = args.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
# torch.backends.cudnn.deterministic = True

globalF_parameters= {}
globalB_parameters= {}
globalC_parameters= {}
# global_parameters = {}  #加载全局模型
netF = network.Net().cuda()
netB = network.feat_bootleneck().cuda()
netC = network.feat_classifier().cuda()
for key,var in netF.state_dict().items():
    globalF_parameters[key] = var.clone()
    # print(var.shape)
for key1,var1 in netB.state_dict().items():
    globalB_parameters[key1] = var1.clone()
for key2,var2 in netC.state_dict().items():
    globalC_parameters[key2] = var2.clone()
source_domain = ['A','E','F']
df =len(source_domain)
for k in range(300):
    print("communicate round {}".format(k + 1))
    sum_parametersF = None
    sum_parametersB = None
    sum_parametersC = None
    for i in source_domain:
        print('source_domain{}'.format(i))
        args.output_dir = osp.join(args.output, 'seed' + str(args.seed) + '{}'.format(i))
        Shot_1 = shot(args, source=i, target='B',globalF_parameters=globalF_parameters,globalB_parameters=globalB_parameters,
                      globalC_parameters=globalC_parameters,k=k)
        # args.output_dir = osp.join(args.output, 'seed' + str(args.seed) + '{}'.format(i))

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        # if not osp.exists(osp.join(args.output_dir + '/source_F.pt')):  #socure shi fou chongxin xunlian
        Shot_1.train_source()
        args.out_file = open(osp.join(args.output_dir, 'log_src.txt'), 'w')
        args.out_file.write(Shot_1.print_args() + '\n')
        args.out_file.flush()

        args.savename = 'par_' + str(args.cls_par)

        net = Shot_1.train_target() #更新网络模型
        # print(net[0].items)
        # print('netF[0].state_dict()', net[0])
        # local_parameters = net[0].state_dict
        # print(local_parameters.items())
        # print('netB[1].state_dict()', net[1].state_dict())
        # print('netC[2].state_dict()', net[2].state_dict())

        if sum_parametersF is None:
            sum_parametersF = {}
            for key, var in net[0].items():
                # print('aaaaaaa',net[0].items())
                sum_parametersF[key] = var.clone()
        else:
            for var in sum_parametersF:
                sum_parametersF[var] = sum_parametersF[var] + net[0][var]

        if sum_parametersB is None:
            sum_parametersB = {}
            for key1, var1 in net[1].items():
                # print('aaaaaaa',net[0].items())
                sum_parametersB[key1] = var1.clone()
        else:
            for var1 in sum_parametersB:
                sum_parametersB[var1] = sum_parametersB[var1] + net[1][var1]

        if sum_parametersC is None:
            sum_parametersC = {}
            for key2, var2 in net[2].items():
                # print('aaaaaaa',net[0].items())
                sum_parametersC[key2] = var2.clone()
        else:
            for var2 in sum_parametersC:
                sum_parametersC[var2] = sum_parametersC[var2] + net[2][var2]

    for var in globalF_parameters:
        globalF_parameters[var] = (sum_parametersF[var] / df )
        # print('sum_parameters[var]',sum_parameters[var].shape)
        # print('globalF_parameters[var]',globalF_parameters[var])

    for var1 in globalB_parameters:
        globalB_parameters[var1] = (sum_parametersB[var1] / df )

    for var2 in globalC_parameters:
        globalC_parameters[var2] = (sum_parametersC[var2] / df )

    with torch.no_grad():
        # print('globalF_parameters', globalF_parameters)
        netF.load_state_dict(globalF_parameters, strict=True)
        # print('globalF_parameters',globalF_parameters)
        netB.load_state_dict(globalB_parameters, strict=True)
        netC.load_state_dict(globalC_parameters, strict=True)
        correct = 0
        total = 0
        c = []
        d = []
        test_loader, train_loader = Get_dataloader()

        for i, q in enumerate(test_loader):
            test_data = Variable(q['B'].type(torch.FloatTensor)).cuda()
            test_label = q['label_B'][:, 0].cuda()
            # print(test_label)
            # net = netC(netB(netF()))
            out = netC(netB(netF(test_data)))
            _, pre = torch.max(out.data, 1)
            c.extend(pre)
            d.extend(test_label)
            correct += (pre == test_label.long()).sum().item()
            total += test_label.size(0)
            # print('total',total)
        acc = correct / total

        # np.save('/home/yq/corss-person/biaoqianjuzhen/pre'+str(k + 1)+'.npy',c)
        # np.save('/home/yq/corss-person/biaoqianjuzhen/zhenshi'+str(k+ 1)+'.npy', d)
        # sys.stdout = Logger('A11.txt')
        print('final target accuracy on the test is:', acc)
        with open('targetB', 'a+') as f:
            f.write(str(acc) +  '\n')
        f.close()






