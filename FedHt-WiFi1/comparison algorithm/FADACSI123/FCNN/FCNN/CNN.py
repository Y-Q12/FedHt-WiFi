from data import Get_dataloader

# from functions import MSE, SIMSE, DiffLoss
import torch.optim as optim
from scipy import spatial
from torch.autograd import Variable
import numpy as np
import random
import torch
import torch.nn as nn
import time
torch.cuda.set_device(3)


#out=128通道,stride=2,三层不行


#就用这个模型（迁移学习）
# def step_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#
# step_seed(12)

cuda = True
lr = 0.001
gamma_weight = 0.35
active_domain_loss_step = 500


#官方程序： mse_loss 0.01  dann_loss 0.25
n_epoch = 200
step_decay_weight = 0.95
lr_decay_step = 20000
weight_decay = 1e-6
alpha_weight = 0.01
beta_weight = 0.075
momentum = 0.9

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
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        class_pre = self.class_d(feature)

        return class_pre

def save_model(model, filename):
    # state = model.state_dict()
    torch.save(model, filename)


class client(object):
    def test_source(epoch, my_net, test_loader, total_time):
        my_net.eval()

        correct = 0
        total = 0
        for i, q in enumerate(test_loader):
            test_data = Variable(q['B'].type(torch.FloatTensor)).cuda()
            test_label = q['label_B'][:, 0].cuda()
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

        total_time = 0
        test_loader, train_loader = Get_dataloader()
        net_S = CNN().cuda()

        optimizer_net_S = optim.Adam(net_S.parameters(), lr=0.001, betas=(0.5, 0.999))
        loss_classification = nn.CrossEntropyLoss()

        if cuda:
            loss_classification = loss_classification.cuda()

        for epoch in range(n_epoch):
            start_time = time.time()
            net_S.train()

            for i, d in enumerate(train_loader):
                source_data = Variable(d['A'].type(torch.FloatTensor)).cuda()

                source_class_l = d['label_A'][:, 0].cuda()
                # train source model
                optimizer_net_S.zero_grad()

                pre = net_S(source_data)

                class_loss = loss_classification(pre, source_class_l.long())
                class_loss.backward()
                optimizer_net_S.step()

                print('[%d/%d][%d/%d] loss: %.4f  ' % (
                    epoch, n_epoch, i, len(train_loader), class_loss))
            end_time = time.time()
            total_time += round((end_time - start_time) / 60, 2)
            print("训练至第{}个epoch耗费时间为： {}分钟".format(epoch, round(total_time, 2)))
            client.test_source(epoch, net_S, test_loader, total_time)
            return net_S

if __name__ == '__main__':
    train()