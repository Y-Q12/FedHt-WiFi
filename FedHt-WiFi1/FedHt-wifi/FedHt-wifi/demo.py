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
from data import Get_dataloader
# def noise(data, snr):
#     Ps = torch.sum(abs(data) ** 2) / len(data)
#     Pn = Ps / (10 ** ((snr / 10)))
#     noise = torch.normal(mean=(len(data)) * np.sqrt(Pn))
#     new_data = data + noise
#     return new_data
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        # self.log = open(filename, "a", encoding="utf-8")  # 防止编码错误

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
class shot(object):
    def __init__(self,args,source,target,globalF_parameters,globalB_parameters,globalC_parameters,k):#,globalF_parameters,globalB_parameters,globalC_parameters
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
        self.globalF_parameters = globalF_parameters
        self.globalB_parameters = globalB_parameters
        self.globalC_parameters = globalC_parameters
        self.k = k
        # print(self.globalF_parameters)

        # print(self.target)
        # self.train_source(args)
        # self.train_target(args)



    def test_target(self,epoch, netF, netB, netC, test_loader):
        epoch =epoch

        netF.eval()
        netB.eval()
        netC.eval()
        # print(netF.eval(),netB.eval(),netC.eval() )# cl_D.eval()
        correct = 0
        total = 0
        # writer = SummaryWriter('yqlog')
        for i, q in enumerate(test_loader):

            test_data = Variable(q['{}'.format(self.target)].type(torch.FloatTensor)).cuda()
            test_label = q['label_{}'.format(self.target)][:, 0].cuda()
            # print(test_label)
            out = netC(netB(netF(test_data)))
            _, pre = torch.max(out.data, 1)
            correct += (pre == test_label.long()).sum().item()
            total += test_label.size(0)
            # print('total',total)
        acc = correct / total
        print('epoch',epoch,'target accuracy on the test is:', acc)

        # self.domain=a

        # writer.add_scalar('jin',acc,3)
        # print('nef', netF.state_dict())
        torch.save(netF.state_dict(), osp.join(self.output_dir, "target_F_" + '{}'.format(self.source)+".pt"))#########
        torch.save(netB.state_dict(), osp.join(self.output_dir, "target_B_"+ '{}'.format(self.source) + ".pt"))
        torch.save(netC.state_dict(), osp.join(self.output_dir, "target_C_"+ '{}'.format(self.source) + ".pt"))

    def test_source(self, epoch, netF, netB, netC, test_loader):
        epoch =epoch
        netF.eval()
        netB.eval()
        netC.eval()  # cl_D.eval()
        correct = 0
        total = 0

        for i, q in enumerate(test_loader):
            # print('aaaaaaaa',self.source)
            test_data = Variable(q['{}'.format(self.source)].type(torch.FloatTensor)).cuda()
            test_label = q['label_{}'.format(self.source)][:, 0].cuda()
            out = netC(netB(netF(test_data)))
            _, pre = torch.max(out.data, 1)
            correct += (pre == test_label.long()).sum().item()
            total += test_label.size(0)
        acc = correct / total
        print('epoch',epoch,'source accuracy on the test is:', acc)
        # if self.k==0:
        f = netB.state_dict()
        for k, v in f.items():
                # f[k] = noise(v,6.02)
                # f[k] = v + np.random.normal(loc=0, scale=0.1)   #gaosi
            f[k] = v+np.random.laplace(loc=0, scale=0.01)  #lapulasi
        # # else:
            # f =netB.state_dict()





        torch.save(netF.state_dict(), osp.join(self.output_dir, "source_F.pt"))
        # print('self.output_dir',self.output_dir)
        torch.save(f, osp.join(self.output_dir, "source_B.pt"))
        torch.save(netC.state_dict(), osp.join(self.output_dir, "source_C.pt"))


    def train_source(self):
        test_loader, train_loader = Get_dataloader()

        netF = network.Net().cuda()
        netB = network.feat_bootleneck().cuda()
        netC = network.feat_classifier().cuda()

        if not self.k ==0:
            print('aaaaaaaaa')
            netF.load_state_dict(self.globalF_parameters, strict=True)
            # print('self.globalF_parameters',self.globalF_parameters)
            netB.load_state_dict(self.globalB_parameters, strict=True)
            # print('self.globalB_parameters',self.globalB_parameters)
            netC.load_state_dict(self.globalC_parameters, strict=True)

        param_group = []
        learning_rate = self.lr
        for k, v in netF.named_parameters():
            param_group += [{'params': v, 'lr': learning_rate}]
        for k, v in netB.named_parameters():
            param_group += [{'params': v, 'lr': learning_rate}]
        for k, v in netC.named_parameters():
            param_group += [{'params': v, 'lr': learning_rate}]

        optimizer = torch.optim.Adam(param_group)
        epoches = self.source_epoch
        # loss_c=nn.CrossEntropyLoss()
        # train_cnt = AverageMeter()
        for epoch in range(epoches):
            netF.train()
            netB.train()
            netC.train()

            # train_loader.reset()
            for i, d in enumerate(train_loader):
                source_data = Variable(d['{}'.format(self.source)].type(torch.FloatTensor)).cuda()
                source_label = d['label_{}'.format(self.source)][:, 0].cuda()
                # print(source_label)
                optimizer.zero_grad()
                outputs_source = netC(netB(netF(source_data)))
                classifier_loss = loss.CrossEntropyLabelSmooth(num_classes=self.class_num, epsilon=self.smooth)(
                    outputs_source, source_label.long())
                # classifier_loss = loss_c(outputs_source, source_label.long())

                classifier_loss.backward()
                optimizer.step()

                # print('[%d/%d][%d/%d]  loss: %.4f   ' % (epoch, epoches, i, len(train_loader), classifier_loss))
            shot.test_source(self, epoch, netF, netB, netC, test_loader)
        return netF, netB, netC

    def print_args(self):
        s = "==========================================\n"
        for arg, content in self.args.__dict__.items():
            s += "{}:{}\n".format(arg, content)
        return s

    def train_target(self):
        test_loader, train_loader = Get_dataloader()
        netF = network.Net().cuda()
        netB = network.feat_bootleneck().cuda()
        netC = network.feat_classifier().cuda()

        self.modelpath = self.output_dir + '/source_F.pt'
        # print('output_dir',self.output_dir)
        netF.load_state_dict(torch.load(self.modelpath))
        # print('netF.load_state_dict',netF.load_state_dict(torch.load(self.modelpath)))
        self.modelpath = self.output_dir + '/source_B.pt'
        netB.load_state_dict(torch.load(self.modelpath))
        self.modelpath = self.output_dir + '/source_C.pt'
        netC.load_state_dict(torch.load(self.modelpath))
        netC.eval()
        for k, v in netC.named_parameters():
            v.requires_grad = False


            param_group = []
            learning_rate = self.lr
            # print(self.lr)



        # # print('netC.eval',netC.eval())
        # for k, v in netC.named_parameters():
        #     v.requires_grad = False

        param_group = []
        for k, v in netF.named_parameters():
            param_group += [{'params': v, 'lr': self.lr}]
        for k, v in netB.named_parameters():
            param_group += [{'params': v, 'lr': self.lr}]

        optimizer = torch.optim.Adam(param_group)
        epoches = self.traget_epoch

        # train_cnt = AverageMeter()
        for epoch in range(epoches):
            netF.train()
            # print('netF.state_dict',netF.state_dict())
            netB.train()
            netC.train()

            # train_loader.reset()
            for i, d in enumerate(train_loader):

                target_data = Variable(d['{}'.format(self.target)].type(torch.FloatTensor)).cuda()
                target_label = d['label_{}'.format(self.target)][:, 0].cuda()
                # print(target_data.shape)

                outputs_test = netC(netB(netF(target_data)))

                mem_label = shot.obtain_label(self,target_data, target_label, netF, netB, netC)
                mem_label = torch.from_numpy(mem_label)

                if self.cls_par > 0:
                    pred = mem_label
                    # print(pred)
                    classifier_loss = self.cls_par * nn.CrossEntropyLoss()(outputs_test.cuda(), pred.long().cuda())
                else:
                    # print(000000000000000)
                    classifier_loss = torch.tensor(0.0).cuda()

                if self.ent:
                    softmax_out = nn.Softmax(dim=1)(outputs_test)
                    entropy_loss = torch.mean(loss.Entropy(softmax_out))
                    if self.gent:
                        msoftmax = softmax_out.mean(dim=0)
                        entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

                    im_loss = entropy_loss * self.ent_par
                    classifier_loss += im_loss
                optimizer.zero_grad()
                classifier_loss.backward()
                optimizer.step()

                # print('[%d/%d][%d/%d]  loss: %.4f   ' % (epoch, epoches, i, len(train_loader), classifier_loss))
            shot.test_target(self, epoch, netF, netB, netC, test_loader,)
        return netF.state_dict(), netB.state_dict(), netC.state_dict()

    def obtain_label(self,loader, lable, netF, netB, netC,  c=None):
        start_test = True
        with torch.no_grad():
            data = loader
            inputs = data
            labels = lable
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            # print(outputs)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                # all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()))
                all_output = torch.cat((all_output, outputs.float().cpu()))
                # all_label = torch.cat((all_label, labels.float()))
        all_output = nn.Softmax(dim=1)(all_output)
        # print(all_output)
        _, predict = torch.max(all_output, 1)

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)

        for round in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea, initc, 'cosine')
            pred_label = dd.argmin(axis=1)

        return pred_label.astype('int')









