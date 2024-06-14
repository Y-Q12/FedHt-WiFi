from __future__ import print_function
import torch
import sys
sys.path.append('./model')
sys.path.append('./datasets')
#
print(sys.path)

import torch.nn as nn
import math
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model.build_gen import *
from datasets.dataset_read import fda_dataset_read
from itertools import cycle, zip_longest
from sklearn.cluster import KMeans
import numpy as np

# Training settings
class Solver(object):
    def __init__(self, args, batch_size=32, source='federated', target='usps', learning_rate=0.0002, interval=100, optimizer='adam'
                 , num_k=4, all_use=False, checkpoint_dir=None, save_epoch=10, gradient_decay_rate=0.1):
        self.src_domain_code = np.repeat(np.array([[*([1]), *([0])]]), batch_size, axis=0)
        self.tgt_domain_code = np.repeat(np.array([[*([0]), *([1])]]), batch_size, axis=0)
        self.src_domain_code = Variable(torch.FloatTensor(self.src_domain_code).cuda(), requires_grad=False)
        self.tgt_domain_code = Variable(torch.FloatTensor(self.tgt_domain_code).cuda(), requires_grad=False)
        self.batch_size = batch_size
        self.source = source
        self.target = target
        self.num_k = num_k
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.use_abs_diff = args.use_abs_diff
        self.all_use = all_use
        self.scale = False
        self.domain_all = ['A', 'B', 'C']
        self.domain_all.remove(target)
        self.interval=interval
        self.lr = learning_rate
        self.softmax = lambda z:np.exp(z)/np.sum(np.exp(z))
        self.gradient_decay_rate = gradient_decay_rate
        self.coefficient_matrix=[0.25]*4
        self.pre_inertia = -1
        self.inertia = [0]*4
        self.root  = '/root/tf-logs/iclr_2020_code_federated_adversarial_domain_adaptation/data/zyydata'


        self.total_iteration = 0

        print('dataset loading')
        self.dataset_s = []
        self.dataset_test_s = []

        for i, domain in enumerate(self.domain_all):
            print('sssssssssssss',self.domain_all[i])
            dataset, dataset_test = fda_dataset_read(self.root,self.domain_all[i], self.batch_size, )
            self.dataset_s.append(dataset)
            self.dataset_test_s.append(dataset_test)
        self.dataset_t, self.dataset_test_t = fda_dataset_read(self.root,target, self.batch_size)
        print('dataset loading finished!')


        print('building models')
        self.G_s = []
        self.C_s = []
        self.FD = []
        self.D = []
        self.DC = []
        self.R = []
        self.M = []
        for i, domain in enumerate(self.domain_all):
            self.G_s.append(Generator(source=self.domain_all[i],target=target))
            self.C_s.append(Classifier(source=self.domain_all[i],target=target))
            self.FD.append(Feature_Discriminator())
            self.D.append(Disentangler())
            self.DC.append(Classifier(source=self.domain_all[i], target=target))
            self.R.append(Reconstructor())
            self.M.append(Mine())
        self.G_t = Generator(source='federated', target=target)
        self.C_t = Classifier(source='federated', target=target)
        print('building models finished')

        for G_s, C_s, FD, D, DC, R, M in zip(self.G_s, self.C_s, self.FD, self.D, self.DC, self.R, self.M):
            G_s.cuda()
            C_s.cuda()
            FD.cuda()
            D.cuda()
            DC.cuda()
            R.cuda()
            M.cuda()

        self.G_t.cuda()
        self.C_t.cuda()

        # setting optimizer
        self.opt_g_s = []
        self.opt_c_s = []
        self.opt_fd = []
        self.opt_d = []
        self.opt_dc = []
        self.opt_r = []
        self.opt_m = []

        for G_s, C_s, FD, D, DC, R, M in zip(self.G_s, self.C_s, self.FD, self.D, self.DC, self.R, self.M):
            self.opt_g_s.append(optim.SGD(G_s.parameters(), lr=learning_rate*50, weight_decay=0.0005))
            self.opt_c_s.append(optim.SGD(C_s.parameters(), lr=learning_rate*50, weight_decay=0.0005))
            self.opt_fd.append(optim.SGD(FD.parameters(), lr=learning_rate*50, weight_decay=0.0005))
            self.opt_d.append(optim.SGD(D.parameters(), lr=learning_rate*50, weight_decay=0.0005))
            self.opt_dc.append(optim.SGD(DC.parameters(), lr=learning_rate*50, weight_decay=0.0005))
            self.opt_r.append(optim.SGD(R.parameters(), lr=learning_rate*50, weight_decay=0.0005))
            self.opt_m.append(optim.SGD(M.parameters(), lr=learning_rate*50, weight_decay=0.0005))

        self.opt_g_t = optim.SGD(self.G_t.parameters(), lr=learning_rate*50, weight_decay=0.0005)
        self.opt_c_t = optim.SGD(self.C_t.parameters(), lr=learning_rate*50, weight_decay=0.0005)

        # initilize parameters
        for G in self.G_s:
            for net, net_cardinal in zip(G.named_parameters(), self.G_t.named_parameters()):
                net[1].data = net_cardinal[1].data.clone()
        for C in self.C_s:
            for net, net_cardinal in zip(C.named_parameters(), self.C_t.named_parameters()):
                net[1].data = net_cardinal[1].data.clone()

    def guassian_kernel(self,source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val) 

    def MK_MMD(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target,
                                       kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        loss = 0
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]
        return loss / float(batch_size)



    def zip_cycle(self, *iterables, empty_default=None):
        cycles = [cycle(i) for i in iterables]
        for _ in zip_longest(*iterables):
            yield tuple(next(i, empty_default) for i in cycles)

    def ent(self, output):
        return - torch.mean(output * torch.log(output + 1e-6))

    def mutual_information_estimator(self, index, x, y, y_):
        joint,marginal = self.M[index](x,y), self.M[index](x,y_)
        return torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal)))

    def reconstruct_loss(self, src, tgt):
        return torch.sum((src-tgt)**2) / (src.shape[0] * src.shape[1])

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1,dim=1) - F.softmax(out2,dim=1)))

    def reset_grad(self):
        for (opt_g_s,opt_c_s,opt_fd, opt_d, opt_dc, opt_m,opt_r) in \
                zip(self.opt_g_s, self.opt_c_s, self.opt_fd, self.opt_d, self.opt_dc, self.opt_m,self.opt_r):
            opt_g_s.zero_grad()
            opt_c_s.zero_grad()
            opt_fd.zero_grad()
            opt_d.zero_grad()
            opt_dc.zero_grad()
            opt_m.zero_grad()
            opt_r.zero_grad()
        self.opt_c_t.zero_grad()
        self.opt_g_t.zero_grad()

    def group_step(self, step_list):
        for i in range(len(step_list)):
            step_list[i].step()
        self.reset_grad()

    def train(self, epoch, record_file=None):

        criterion=nn.CrossEntropyLoss().cuda()
        adv_loss = nn.BCEWithLogitsLoss().cuda()
        torch.cuda.manual_seed(1)
        for G, C, FD, D, DC, R, M in zip(self.G_s, self.C_s, self.FD, self.D, self.DC, self.R, self.M):
            G.train()
            C.train()
            FD.train()
            D.train()
            DC.train()
            R.train()
            M.train()
        self.G_t.train()
        self.C_t.train()

        test_batch = next(iter(self.dataset_t))

        for batch_idx, (data_s1, data_s2,  data_t) in \
                enumerate(self.zip_cycle(self.dataset_s[0],self.dataset_s[1],
                                         
                                         self.dataset_t)):
        
            # print('ccccccccc',data_t['img'].size()[0])
            # if data_s1['img'].size()[0]<self.batch_size or data_s2['img'].size()[0]<self.batch_size\
            #     or data_s3['img'].size()[0]<self.batch_size or data_s4['img'].size()[0]<self.batch_size\
            #     or data_t['img'].size()[0]<self.batch_size:
            #     break
            data_all = [data_s1, data_s2]
            self.total_iteration += 1
            loss_record = []
            loss_t_record = []
            for i, data in enumerate(data_all):
                img, img_t = data['img'], data_t['img']
                label, label_t = data['label'], data_t['label']
                img, img_t = Variable(img.cuda()), Variable(img_t.cuda())
                label, label_t = Variable(label.long().cuda()), Variable(label_t.long().cuda())
                self.reset_grad()
                feat = self.G_s[i](img)
                feat_fc2 = feat['f_fc2']
                output = self.C_s[i](feat_fc2)

                feat_t = self.G_t(img_t)
                feat_fc2_t = feat_t['f_fc2']
                output_t = self.C_t(feat_fc2_t)

                loss = criterion(output, label.squeeze(dim=1))
                loss_t = criterion(output_t, label_t.squeeze(dim=1))
                loss_record.append(loss.data)
                loss_t_record.append(loss_t.data)

                loss.backward(retain_graph=True)




                feat_conv3 = feat['f_conv3']
                output_d=self.DC[i](self.D[i](feat_conv3))
                loss = criterion(output, label.squeeze(dim=1))
                loss_dis = self.discrepancy(output,output_d)
                loss -= loss_dis
                loss.backward(retain_graph=True)


                for net1, net2 in zip(self.G_s[i].named_parameters(), self.G_t.named_parameters()):
                    if net1[1].grad is not None:
                        net2[1].grad = net1[1].grad.clone() * self.coefficient_matrix[i] 
                for net1, net2 in zip(self.C_s[i].named_parameters(), self.C_t.named_parameters()):
                    if net1[1].grad is not None:
                        net2[1].grad = net1[1].grad.clone() * self.coefficient_matrix[i] 
                self.group_step([self.opt_d[i], self.opt_dc[i], self.opt_g_t, self.opt_c_t, self.opt_g_s[i], self.opt_c_s[i]])

                features_list = self.G_s[i](img)
                dis_features = self.D[i](features_list['f_conv3'])
                dis_features_shuffle = torch.index_select(dis_features,0,Variable(torch.randperm(dis_features.shape[0]).cuda()))
                features_fc2 = features_list['f_fc2']
                MI = self.mutual_information_estimator(i, features_fc2, dis_features, dis_features_shuffle) / self.batch_size
                entropy_s = self.ent(dis_features) / self.batch_size
                recon_features = self.R[i](torch.cat((features_fc2, dis_features),1)) / self.batch_size
                recon_loss =  self.reconstruct_loss(features_list['f_conv3'], recon_features) / self.batch_size
                total_loss = (MI - entropy_s + recon_loss ) * 0.1
                total_loss.backward()
                self.group_step([self.opt_d[i], self.opt_r[i], self.opt_m[i], self.opt_g_s[i]])


                img =test_batch['img']
                img = Variable(img.cuda())
                feat_torch = self.G_t(img)['f_fc2']

                feat = feat_torch.data.cpu().numpy()

                kmeans = KMeans(n_clusters=10, max_iter=1000,n_init='auto').fit(feat)


                cur_inertia = kmeans.inertia_ / 1e3
                if self.pre_inertia == -1:
                    self.pre_inertia = cur_inertia
                    continue
                else:
                    inertia_gain = (self.pre_inertia - cur_inertia)/self.coefficient_matrix[i]*0.25
                    self.inertia[i] = inertia_gain + self.inertia[i]
                    self.pre_inertia = cur_inertia

                src_domain_pred = self.FD[i](self.G_s[i](img)['f_fc2'])
                tgt_domain_pred = self.FD[i](self.G_t(img_t)['f_fc2'])
                # print('aaaaaa',img.shape)
                # print(tgt_domain_pred.shape)
                # print(self.tgt_domain_code.shape)

                df_loss_src = adv_loss(src_domain_pred, self.src_domain_code)
                df_loss_tgt = adv_loss(tgt_domain_pred, self.tgt_domain_code)

                loss = (df_loss_src + df_loss_tgt)/self.batch_size
                loss.backward()
                self.group_step([self.opt_fd[i], self.opt_g_s[i], self.opt_g_t])

                src_domain_pred = self.FD[i](self.G_s[i](img)['f_fc2'])
                # print(src_domain_pred)
                tgt_domain_pred = self.FD[i](self.G_t(img_t)['f_fc2'])

                df_loss_src = adv_loss(src_domain_pred, 1-self.src_domain_code)
                df_loss_tgt = adv_loss(tgt_domain_pred, 1-self.tgt_domain_code)
                loss = (df_loss_src + df_loss_tgt)/self.batch_size
                loss.backward()
                self.group_step([self.opt_fd[i], self.opt_g_s[i], self.opt_g_t])




            coefficient_matrix_diff = self.softmax(self.inertia)
            self.coefficient_matrix = [0.20 + 0.2 * tmp for tmp in coefficient_matrix_diff]
            if record_file:
                record = open(record_file, 'a')
                record.write('{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}   \n'.format(
                    self.coefficient_matrix[0], self.coefficient_matrix[1], self.coefficient_matrix[2], self.coefficient_matrix[3],
                    loss_record[0], loss_record[1], 
                    
                    loss_t_record[0], loss_t_record[1]
                    ))
                record.close()

        return batch_idx

    def test(self, epoch, record_file=None, save_model=False):
     with torch.no_grad():
        self.G_t.eval()
        self.C_t.eval()
        test_loss = 0
        size = 0
        correct = 0
        for batch_idx, data in enumerate(self.dataset_test_t):
            img = data['img']
            label = data['label']
            img, label = img.cuda(), label.long().cuda()
            # img, label = Variable(img, volatile=True), Variable(label)
            feat  = self.G_t(img)['f_fc2']
            output = self.C_t(feat)
            # print('output',output.shape)
            test_loss += F.nll_loss(output, label.squeeze(dim=1)).data
            _,pred = torch.max(output.data, 1)
            # print(pred.shape)
            # print(label.squeeze(dim=1).shape)
            correct += (pred == label.squeeze(dim=1)).sum().item()
            k = label.data.size()[0]
            size += k
        acc = correct/size
        print(acc)
            
        #     k = label.data.size()[0]
        #     correct += pred.eq(label.data).cpu().sum()
        #     size += k
        # test_loss = test_loss / size
        # print('\nTest_set: Average loss: {:.4f}, Accuracy C:{}/{} ({:.2f}%) \n'.format(
        #     test_loss, correct, size, correct/size
        # ))
        # # print('aaaaaa',correct)
        # if record_file:
        #     record = open(record_file, 'a')
        #     record.write('Average loss: {:.4f}, Accuracy: {}/{} {:.2f}% \n'.format(
        #         test_loss, correct, size, 100*correct/size))
        #     record.close()


