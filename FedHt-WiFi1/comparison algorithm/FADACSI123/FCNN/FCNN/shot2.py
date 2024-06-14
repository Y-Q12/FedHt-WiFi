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
from data import Get_dataloader


def test_target(args, epoch, netF, netB, netC, test_loader):
    netF.eval()
    netB.eval()
    netC.eval()  # cl_D.eval()
    correct = 0
    total = 0

    for i, q in enumerate(test_loader):
        test_data = Variable(q['B'].type(torch.FloatTensor)).cuda()
        test_label = q['label_B'][:, 0].cuda()
        out = netC(netB(netF(test_data)))
        _, pre = torch.max(out.data, 1)
        correct += (pre == test_label.long()).sum().item()
        total += test_label.size(0)
    acc = correct / total
    print('target accuracy on the test is:', acc)

    torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + ".pt"))
    torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + ".pt"))
    torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + ".pt"))


def test_source(args, epoch, netF, netB, netC, test_loader):
    netF.eval()
    netB.eval()
    netC.eval()  # cl_D.eval()
    correct = 0
    total = 0

    for i, q in enumerate(test_loader):
        test_data = Variable(q['A'].type(torch.FloatTensor)).cuda()
        test_label = q['label_A'][:, 0].cuda()
        out = netC(netB(netF(test_data)))
        _, pre = torch.max(out.data, 1)
        correct += (pre == test_label.long()).sum().item()
        total += test_label.size(0)
    acc = correct / total
    print('source accuracy on the test is:', acc)

    torch.save(netF.state_dict(), osp.join(args.output_dir, "source_F.pt"))
    torch.save(netB.state_dict(), osp.join(args.output_dir, "source_B.pt"))
    torch.save(netC.state_dict(), osp.join(args.output_dir, "source_C.pt"))


def train_source(args):
    test_loader, train_loader = Get_dataloader()
    netF = network.Net().cuda()
    netB = network.feat_bootleneck().cuda()
    netC = network.feat_classifier().cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]

    optimizer = torch.optim.Adam(param_group)
    epoches = 10000
    # loss_c=nn.CrossEntropyLoss()
    # train_cnt = AverageMeter()
    for epoch in range(epoches):
        netF.train()
        netB.train()
        netC.train()

        # train_loader.reset()
        for i, d in enumerate(train_loader):
            source_data = Variable(d['A'].type(torch.FloatTensor)).cuda()
            source_label = d['label_A'][:, 0].cuda()
            # print(source_label)
            optimizer.zero_grad()
            outputs_source = netC(netB(netF(source_data)))
            classifier_loss = loss.CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(
                outputs_source, source_label.long())
            # classifier_loss = loss_c(outputs_source, source_label.long())


            classifier_loss.backward()
            optimizer.step()

            print('[%d/%d][%d/%d]  loss: %.4f   ' % (epoch, epoches, i, len(train_loader), classifier_loss))
        test_source(args, epoch, netF, netB, netC, test_loader)
    return netF, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def train_target(args):
    test_loader, train_loader = Get_dataloader()
    netF = network.Net().cuda()
    netB = network.feat_bootleneck().cuda()
    netC = network.feat_classifier().cuda()
    param_group = []
    learning_rate = args.lr

    args.modelpath = args.output_dir + '/source_F.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_B.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]

    optimizer = torch.optim.Adam(param_group)
    epoches = 10000

    # train_cnt = AverageMeter()
    for epoch in range(epoches):
        netF.train()
        netB.train()
        netC.train()

        # train_loader.reset()
        for i, d in enumerate(train_loader):

            target_data = Variable(d['B'].type(torch.FloatTensor)).cuda()
            target_label = d['label_B'][:, 0].cuda()

            outputs_test = netC(netB(netF(target_data)))

            mem_label = obtain_label(target_data, target_label, netF, netB, netC, args)
            mem_label = torch.from_numpy(mem_label)
            print(mem_label)
            if args.cls_par > 0:
                pred = mem_label
                classifier_loss = args.cls_par * nn.CrossEntropyLoss()(outputs_test.cuda(), pred.long().cuda())

            if args.ent:
                softmax_out = nn.Softmax(dim=1)(outputs_test)
                entropy_loss = torch.mean(loss.Entropy(softmax_out))
                if args.gent:
                    msoftmax = softmax_out.mean(dim=0)
                    entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

                im_loss = entropy_loss * args.ent_par
                classifier_loss += -im_loss
            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()

            print('[%d/%d][%d/%d]  loss: %.4f   ' % (epoch, epoches, i, len(train_loader), classifier_loss))
        test_target(args, epoch, netF, netB, netC, test_loader)
    return netF, netB, netC


def obtain_label(loader, lable, netF, netB, netC, args, c=None):
    start_test = True
    with torch.no_grad():
        data = loader
        inputs = data
        labels = lable
        inputs = inputs.cuda()
        feas = netB(netF(inputs))
        outputs = netC(feas)
        print(outputs)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='2', help="device id to run")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--cls_par', type=float, default=1)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()
    args.class_num = 7

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    args.output_dir = osp.join(args.output, 'seed' + str(args.seed))
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if not osp.exists(osp.join(args.output_dir + '/source_F.pt')):
        args.out_file = open(osp.join(args.output_dir, 'log_src.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_source(args)

    args.savename = 'par_' + str(args.cls_par)
    args.out_file = open(osp.join(args.output_dir, 'log_tar_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    train_target(args)