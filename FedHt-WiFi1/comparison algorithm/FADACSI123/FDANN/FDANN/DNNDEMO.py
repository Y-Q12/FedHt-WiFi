import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
from data import Get_dataloader
from torch.autograd import Function
import time
torch.cuda.set_device(0)


class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

#AC MSEloss 调完学习率，调特征提取参数大小，通道数, 各项参数不再动修改
class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.feature = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(True),
                                      nn.MaxPool2d(kernel_size=4),
                                   nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(True),
                                   nn.MaxPool2d(kernel_size=4),



                                     )
        self.class_d = nn.Sequential(nn.Linear(8000, 256),
                                   nn.ReLU(True),
                                   nn.Linear(256, 7))
        self.domain_d = nn.Sequential(nn.Linear(8000, 256),
                                   nn.ReLU(True),
                                   nn.Linear(256, 2))
        self.GRL = GRL()

    def forward(self, x, alpha):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        class_pre = self.class_d(feature)
        feature1 = GRL.apply(feature, alpha)
        domain_pre = self.domain_d(feature1)
        return class_pre, domain_pre


class shot(object):
    def __init__(self,args,source,target,globalF_parameters, k):#,globalF_parameters,globalB_parameters,globalC_parameters
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
        self.k = k
        self.globalF_parameters=globalF_parameters
        # print(self.globalF_parameters)

        # print(self.target)
        # self.train_source(args)
        # self.train_target(args)

    def test_target(self,epoch, net, test_loader, total_time):
        net.eval()
        # cl_D.eval()
        correct = 0
        total = 0

        pre_l = []
        for i, q in enumerate(test_loader):
            test_data = Variable(q['{}'.format(self.target)].type(torch.FloatTensor)).cuda()
            test_label = q['label_{}'.format(self.target)][:, 0].cuda()
            out, _ = net(test_data, 0)
            _, pre = torch.max(out.data, 1)
            correct += (pre == test_label.long()).sum().item()
            total += test_label.size(0)

        acc = correct / total
        print(' target accuracy on the test is:', acc)
        # if acc > best:
        #     best=acc
        #     torch.save(net.state_dict(), 'DANN_AB'+'.pkl')#使用model.module.state_dict()
        #     # save_model(net, './DANN_model/DANN_AB'+'.pkl' )
        #     print('best acc', best)
        # with open('DANN_ABtarget.txt', 'a+') as f:
        #     f.write(str(acc) + str(-epoch) + str('-训练至这次耗费总时间为： ') + str(total_time) + '分钟' + '\n')
        # f.close()

    def test_source(self,epoch, net, test_loader, total_time):
        net.eval()
        # cl_D.eval()
        correct = 0
        total = 0
        for i, q in enumerate(test_loader):
            test_data = Variable(q['{}'.format(self.source)].type(torch.FloatTensor)).cuda()
            test_label = q['label_{}'.format(self.source)][:, 0].cuda()
            out, _ = net(test_data, 0)
            _, pre = torch.max(out.data, 1)
            correct += (pre == test_label.long()).sum().item()
            total += test_label.size(0)
        acc = correct / total
        print('source accuracy on the test is:', acc)
        # with open('DANN_model/DANN_BCsource.txt', 'a+') as f:
        #     f.write(str(acc) + str(-epoch) + '\n')
        # f.close()

    def train(self):
        test_loader, train_loader = Get_dataloader()
        net = DANN().cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
        criterion_doamin = torch.nn.CrossEntropyLoss()
        criterion_class = torch.nn.CrossEntropyLoss().cuda()
        epoches = self.traget_epoch
        total_time = 0
        if not self.k == 0:
            net.load_state_dict(self.globalF_parameters,strict=True)

        # train_cnt = AverageMeter()
        for epoch in range(epoches):
            net.train()
            # train_cnt.reset()
            start_time = time.time()
            for i, d in enumerate(train_loader):
                # train_cnt.update(d['A'].size(0), 1)
                p = float(i + epoch * len(train_loader)) / (epoches * len(train_loader))
                alpha = torch.tensor(2. / (1. + np.exp(-10 * p)) - 1)
                correct = 0
                source_data = Variable(d['{}'.format(self.source)].type(torch.FloatTensor)).cuda()
                source_label = d['label_{}'.format(self.source)][:, 0].cuda()

                source_domain_label = Variable(torch.Tensor(d['{}'.format(self.source)].size(0), 1).fill_(1), requires_grad=False).cuda()
                source_domain_label = source_domain_label[:, 0]
                target_doman_label = Variable(torch.Tensor(d['{}'.format(self.target)].size(0), 1).fill_(0), requires_grad=False).cuda()
                target_doman_label = target_doman_label[:, 0]
                # source_domain_label = d['domain_B'].type(torch.FloatTensor).cuda()
                # target_doman_label = d['domain_A'].type(torch.FloatTensor).cuda()

                target_data = Variable(d['{}'.format(self.target)].type(torch.FloatTensor)).cuda()
                # print(target_data.size())

                # train G
                optimizer.zero_grad()

                src_class_pre, src_doamin_pre = net(source_data, alpha)
                # print(src_class_pre.shape)
                # print(source_label.long().shape)
                src_class_loss = criterion_class(src_class_pre, source_label.long())
                src_domain_loss = criterion_doamin(src_doamin_pre, source_domain_label.long())

                _, tar_domain_pre = net(target_data, alpha)
                tar_domain_loss = criterion_doamin(tar_domain_pre, target_doman_label.long())

                loss = src_class_loss + src_domain_loss + tar_domain_loss

                loss.backward()
                optimizer.step()

                # print(
                #     '[%d/%d][%d/%d]  loss: %.4f  source_class_loss: %.4f  src_domain_loss: %.4f  tar_domain_loss: %.4f' % (
                #         epoch, epoches, i, len(train_loader), loss, src_class_loss, src_domain_loss, tar_domain_loss
                #     ))
            end_time = time.time()
            total_time += round((end_time - start_time) / 60, 2)
            print("训练至第{}个epoch耗费时间为: {}分钟".format(epoch, round(total_time, 2)))
            shot.test_source(self,epoch, net, test_loader, total_time)
            shot.test_target(self,epoch, net, test_loader, total_time)

        return net.state_dict()
