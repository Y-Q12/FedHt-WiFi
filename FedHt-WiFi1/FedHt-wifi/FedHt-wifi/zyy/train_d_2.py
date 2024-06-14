from SemiGAN.DSN.datasetspy import Get_dataloader
from SemiGAN.DSN.mode_res1 import Transfer_learning, domain_D
from SemiGAN.DSN.functions import MSE, SIMSE, DiffLoss
import torch.optim as optim
from scipy import spatial
from torch.autograd import Variable
import numpy as np
import random
import torch
import torch.nn as nn
import time
torch.cuda.set_device(3)

#3号卡 0.0001

#out=128通道,stride=2,三层不行


# #就用这个模型（迁移学习）
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
n_epoch = 10000
step_decay_weight = 0.95
lr_decay_step = 20000
weight_decay = 1e-6
alpha_weight = 0.01
beta_weight = 0.075
momentum = 0.9


def save_model(model, filename):
    # state = model.state_dict()
    torch.save(model, filename)

def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos#规划到[0,1]内
    return sim

def k_loss(feature, label):
    loss = 0
    p_1 = 0
    p_2 = 0
    loss_1 = 0
    loss_2 = 0
    for i in range(len(feature)):
        feat_1 = feature[i]
        label_1 = label[i]
        # print('feat_1:', feat_1, label_1)
        for j in range(len(feature)):
            if j > i:
                feat_2 = feature[j]
                label_2 = label[j]

                if int(label_1) == int(label_2):
                    # loss -= feat_dot / feat_sqrt
                    #余弦距离 = 1-余弦相似度，论文中给的是余弦相似度
                    # same_dis = (1 - spatial.distance.cosine(feat_1.cpu().detach().numpy(), feat_2.cpu().detach().numpy()))
                    same_dis = cos_sim(feat_1.cpu().detach().numpy(), feat_2.cpu().detach().numpy())
                    loss_1 += same_dis
                    p_1 += 1
                    # print('smae label consine distance:', same_dis)
                else:
                    # loss += feat_dot / feat_sqrt
                    # different_dis = (1 - spatial.distance.cosine(feat_1.cpu().detach().numpy(), feat_2.cpu().detach().numpy()))
                    different_dis = cos_sim(feat_1.cpu().detach().numpy(), feat_2.cpu().detach().numpy())
                    loss_2 += different_dis
                    p_2 += 1
                    # print('different label consine distance:', different_dis)
            else:
                continue

    to_loss = loss_2 - loss_1
    loss = to_loss / (p_1 + p_2)
    if loss > 0:
        return loss
    else:
        loss = 0.5
        return loss

def test_source(epoch, my_net, test_loader,total_time):
    my_net.eval()
    correct = 0
    total = 0
    for i, q in enumerate(test_loader):
        test_data = Variable(q['A'].type(torch.FloatTensor)).cuda()
        test_label = q['label_A'][:, 0].cuda()
        _,_,_, _,_,out, _, _ = my_net(input_data=test_data, mode='source', rec_scheme='share', mode_1='signal', s_t_data=None)
        # out = result[3]
        _, pre = torch.max(out.data, 1)
        correct += (pre == test_label.long()).sum().item()
        total += test_label.size(0)
    acc = correct / total
    print('source accuracy on the test is:', acc)
    with open('acc_3/source_r14_256.txt', 'a+') as f:
        f.write(str(acc) + str(-epoch)+'-训练至这次耗费总时间为： '+str(total_time)+'分钟' + '\n')
    f.close()

def test_target(epoch, my_net, test_all_loader,total_time):
    my_net.eval()
    correct = 0
    total = 0
    pt = 0
    for i, q in enumerate(test_all_loader):
        pt += 1
        test_data = Variable(q['B'].type(torch.FloatTensor)).cuda()
        test_label = q['label_B'][:, 0].cuda()
        _,_,_, _,_,out, _, _ = my_net(input_data=test_data, mode='target', rec_scheme='share', mode_1='signal', s_t_data=None)
        # out = result[3]
        _, pre = torch.max(out.data, 1)
        correct += (pre == test_label.long()).sum().item()
        total += test_label.size(0)
    print("total num: ", pt)
    acc = correct / total
    print('target accuracy on the test is:', acc)

    # if acc >= 0.12:
    #     save_model(my_net, 'model/net_A-B_res6.pkl')
    with open('acc_3/target_r14_256.txt', 'a+') as f:
        f.write(str(acc) + str(-epoch) +str('-训练至这次耗费总时间为： ')+str(total_time)+'分钟'  + '\n')
    f.close()


def train():

    total_time = 0
    train_loader, test_loader = Get_dataloader()
    # train_loader, test_loader = Get_dataloader()
    my_net = Transfer_learning().cuda()
    do_D = domain_D().cuda()

    optimizer_my_net = optim.Adam(my_net.parameters(), lr=0.0001,betas=(0.5, 0.999))
    optimizer_do_D = optim.Adam(do_D.parameters(), lr=0.0001, betas=(0.5, 0.999))
    # optimizer_my_net = optim.SGD(my_net.parameters(), lr=0.01, momentum=0.9)
    # optimizer_do_D = optim.SGD(do_D.parameters(), lr=0.01, momentum=0.9)

    loss_classification = nn.CrossEntropyLoss()
    loss_recon1 = MSE()
    loss_recon2 = SIMSE()
    loss_diff = DiffLoss()
    loss_similarity = nn.MSELoss()

    if cuda:
        my_net = my_net.cuda()
        loss_classification = loss_classification.cuda()
        loss_recon1 = loss_recon1.cuda()
        loss_recon2 = loss_recon2.cuda()
        loss_diff = loss_diff.cuda()
        loss_similarity = loss_similarity.cuda()

    for p in my_net.parameters():
        p.requires_grad = True

    for epoch in range(n_epoch):
        start_time = time.time()
        my_net.train()
        do_D.train()
        for i, d in enumerate(train_loader):
            ###################################
            # target data training            #
            ###################################

            target_data = Variable(d['B'].type(torch.FloatTensor)).cuda()
            source_data = Variable(d['A'].type(torch.FloatTensor)).cuda()
            source_class_la = d['label_A'].cuda()
            source_class_l = d['label_A'][:, 0].cuda()
            # print(target_data.size())
            target_domain_l = Variable(torch.Tensor(d['B'].size(0), 1).fill_(1), requires_grad=False).cuda()
            # target_domain_l = target_domain_l[:, 0]
            source_domain_l = Variable(torch.Tensor(d['A'].size(0), 1).fill_(0), requires_grad=False).cuda()
            # source_domain_l = source_domain_l[:, 0]

            optimizer_my_net.zero_grad()

            source_private_code, _,_,_,source_shared_code, source_class_label, source_shared_feat, source_rec_data = my_net(source_data,
                                                                                                     mode='source',
                                                                                                     rec_scheme='all',
                                                                                                     mode_1='signal',
                                                                                                     s_t_data=None)

            target_private_code,_,_,_,target_shared_code, target_class_label,  target_shared_feat, target_rec_data = my_net(target_data,
                                                                                                  mode='target',
                                                                                                  rec_scheme='all',
                                                                                                  mode_1='signal',
                                                                                                  s_t_data=None)

            source_domain_label = do_D(source_shared_feat)
            target_domain_label = do_D(target_shared_feat)

            #domain loss
            source_dann_loss = loss_similarity(source_domain_label, target_domain_l)
            target_dann_loss = loss_similarity(target_domain_label, source_domain_l)
            dann_loss = (source_dann_loss + target_dann_loss) / 2

            #source class loss
            kcnn_loss = k_loss(source_shared_code, source_class_la)
            # kcnn_loss = 0
            source_class_loss = loss_classification(source_class_label, source_class_l.long())

            #different loss
            source_diff_loss = loss_diff(source_private_code, source_shared_code)
            target_diff_loss = loss_diff(target_private_code, target_shared_code)
            diff_loss = (source_diff_loss + target_diff_loss) / 2

            #decoder mse and simse loss
            source_mse_loss = loss_recon1(source_rec_data, source_data)
            target_mse_loss = loss_recon1(target_rec_data, target_data)
            mse_loss = (source_mse_loss + target_mse_loss) / 2
            # mse_loss = 0

            #source_target_data
            # rec_s_t_data = source_shared_code + target_private_code
            # s_t_class_label = my_net(input_data=None, mode='source', rec_scheme='all', mode_1='union', s_t_data=rec_s_t_data)
            # s_t_class_loss = loss_classification(s_t_class_label, source_class_l.long())

            source_simse_loss = loss_recon2(source_rec_data, source_data)
            target_simse_loss = loss_recon2(target_rec_data, target_data)
            simse_loss = (source_simse_loss + target_simse_loss) / 2

            #total loss  decoder fc
            loss = source_class_loss +  0.1 * kcnn_loss +  dann_loss + 0.07 * diff_loss + 0.001*mse_loss + 0.001*simse_loss    #speed 12   datasets speed 12  batch 30
            loss.backward()
            optimizer_my_net.step()

            optimizer_do_D.zero_grad()
            _,_,_, _,_, _, s_shared_feat, _ = my_net(source_data, mode='source', rec_scheme='all', mode_1='signal', s_t_data=None)
            _,_,_, _,_,_, t_shared_feat, _ = my_net(target_data, mode='target', rec_scheme='all', mode_1='signal', s_t_data=None)


            s_domain_label = do_D(s_shared_feat)
            t_domain_label = do_D(t_shared_feat)

            s_domain_loss = loss_similarity(s_domain_label, source_domain_l)
            t_domain_loss = loss_similarity(t_domain_label, target_domain_l)

            d_loss = (s_domain_loss + t_domain_loss) / 2
            d_loss.backward()
            optimizer_do_D.step()



            print('[%d/%d][%d/%d] loss: %.4f  source class: %.4f  kcnn_loss: %.4f  diff loss: %.4f  mse_loss: %.4f  simse_loss: %.4f   s_d loss: %.4f  t_d loss: %.4f  domain_D loss: %.4f'%(
                epoch, n_epoch, i, len(train_loader),loss, source_class_loss, kcnn_loss,diff_loss, mse_loss,simse_loss,source_dann_loss,target_dann_loss,d_loss
            ))
        end_time = time.time()
        total_time += round((end_time - start_time) / 60, 2)
        print("训练至第{}个epoch耗费时间为： {}分钟".format(epoch, round(total_time, 2)))
        test_source(epoch, my_net, test_loader, total_time)
        test_target(epoch, my_net, test_loader, total_time)

if __name__ == '__main__':
    train()