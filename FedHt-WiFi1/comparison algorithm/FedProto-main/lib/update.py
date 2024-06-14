#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import glob
import random
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import torch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
from models import CNNFemnist
class ImageDataset(Dataset):

    def __init__(self, root, transforms_=None, unaligned=False, domain='domain', mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, 'train_%s' % domain) + '/*.*'))
        # print('aaaaa',self.files_A)

    def __getitem__(self, index):
        A = np.load(self.files_A[index % len(self.files_A)])

        dat_A = A[:, 1:].astype(np.float32)
        # print('asd',dat_A.shape)
        domain_A = A[0, :1]
        # label_A = A[0, 1:2]
        label_A = A[0, :1].astype(np.float32)
        # print('label_A',label_A)  #zyydata hao

        data_A = np.expand_dims(dat_A, 0)
        # print('aaa',data_A.shape)

        return data_A, label_A

    def __len__(self):
        return len(self.files_A)


class ImageDataset_test(Dataset):

    def __init__(self, root, transforms_=None, unaligned=False, domain='domain', mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, 'test_%s' % domain) + '/*.*'))
        # print(self.files_A)

    def __getitem__(self, index):
        A = np.load(self.files_A[index % len(self.files_A)])

        dat_A = A[:, 1:].astype(np.float32)
        # print('asd',dat_A.shape)
        domain_A = A[0, :1]
        # label_A = A[0, 1:2]
        label_A = A[0, :1].astype(np.float32)
        # print('label_A',label_A)  #zyydata hao

        data_A = np.expand_dims(dat_A, 0)
        # print('aaa',data_A.shape)

        return data_A, label_A
    def __len__(self):
        return len(self.files_A)

def fda_dataset_read(domain):
        # train_loader_1 = DataLoader(ImageDataset('./datasets_1/', unaligned=True, mode='train'),
        #                           batch_size=120, shuffle=True)
        # dataset_test = DataLoader(ImageDataset_test(root, domain=domain, unaligned=True),
        #                           batch_size=batch_size, shuffle=True, drop_last=True)
        dataset = DataLoader(ImageDataset('/home/yq/yqdata/fedpro/', domain=domain, unaligned=True),
                             batch_size=64, shuffle=True, drop_last=False)
        return dataset


def fda_dataset_read1(domain):
    # train_loader_1 = DataLoader(ImageDataset('./datasets_1/', unaligned=True, mode='train'),
    #                           batch_size=120, shuffle=True)
    # dataset_test = DataLoader(ImageDataset_test(root, domain=domain, unaligned=True),
    #                           batch_size=batch_size, shuffle=True, drop_last=True)
    dataset = DataLoader(ImageDataset_test('/home/yq/yqdata/fedpro/', domain=domain, unaligned=True),
                         batch_size=32, shuffle=True, drop_last=False)
    return dataset










class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        # print('ffffssssfffff',len(self.idxs))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # print('546553160',image.shape)
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader = self.train_val_test(dataset, list(idxs))


        self.device = args.device
        self.criterion = nn.NLLLoss().to(self.device)



    def train_val_test(self, dataset, idxs):
        # print('nininininininininininininin',idxs)
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        idxs_train = idxs[:int(1 * len(idxs))]
        # print('hhhhhhhhhhhhhhhhhhhh',idxs_train)
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True, drop_last=True)

        return trainloader

    def update_weights(self, idx, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                log_probs, protos = model(images)
                loss = self.criterion(log_probs, labels)

                loss.backward()
                optimizer.step()

                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))


        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_val.item()

    def update_weights_prox(self, idx, local_weights, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []
        # print('6666666666666666666',local_weights)
        if idx in local_weights.keys():
            w_old = local_weights[idx]
        w_avg = model.state_dict()
        loss_mse = nn.MSELoss().to(self.device)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                log_probs, protos = model(images)
                loss = self.criterion(log_probs, labels)
                if idx in local_weights.keys():
                    loss2 = 0
                    for para in w_avg.keys():
                        loss2 += loss_mse(w_avg[para].float(), w_old[para].float())
                    loss2 /= len(local_weights)
                    loss += loss2 * 150
                loss.backward()
                optimizer.step()

                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))


        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_val.item()

    def update_weights_het(self, args, idx, global_protos, model, global_round=round):
        # Set mode to train model
        model.train()
        epoch_loss = {'total':[],'1':[], '2':[], '3':[]}

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            # print('ffffffffffffffffffffffffffffffffffffffffffffff',idx)
            batch_loss = {'total':[],'1':[], '2':[], '3':[]}
            agg_protos_label = {}
            for batch_idx, (images, label_g) in enumerate(fda_dataset_read(domain=idx)):
                images, labels = images.to(self.device), label_g.to(self.device)
                # print(images.shape)
                # loss1: cross-entrophy loss, loss2: proto distance loss
                model.zero_grad()
                log_probs, protos = model(images)
                labels = labels.long()
                # protos(protos.shape)
                # print('1111111111',labels.shape)
                # print('2222222222',log_probs.shape)
                loss1 = self.criterion(log_probs, labels.squeeze())

                loss_mse = nn.MSELoss()
                if len(global_protos) == 0:
                    loss2 = 0*loss1
                else:
                    proto_new = copy.deepcopy(protos.data)
                    i = 0
                    for label in labels:
                        if label.item() in global_protos.keys():
                            proto_new[i, :] = global_protos[label.item()][0].data
                        i += 1
                    loss2 = loss_mse(proto_new, protos)

                loss = loss1 + loss2 * args.ld
                loss.backward()
                optimizer.step()

                for i in range(len(labels)):
                    if label_g[i].item() in agg_protos_label:
                        agg_protos_label[label_g[i].item()].append(protos[i,:])
                    else:
                        agg_protos_label[label_g[i].item()] = [protos[i,:]]

                log_probs = log_probs[:, 0:args.num_classes]
                _, y_hat = log_probs.max(1)
                # print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',y_hat.shape,labels.squeeze().shape)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                # if self.args.verbose and (batch_idx % 10 == 0):
                #     print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                #         global_round, idx, iter, batch_idx * len(images),
                #         len(self.trainloader.dataset),
                #         100. * batch_idx / len(self.trainloader),
                #         loss.item(),
                #         acc_val.item()))
                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
            epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])

        return model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss

class LocalTest(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.testloader = self.test_split(dataset, list(idxs))
        self.device = args.device
        self.criterion = nn.NLLLoss().to(args.device)

    def test_split(self, dataset, idxs):
        idxs_test = idxs[:int(1 * len(idxs))]

        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                 batch_size=64, shuffle=False)
        return testloader

    def get_result(self, args, idx, classes_list, model):
        # Set mode to train model
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            model.zero_grad()
            outputs, protos = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            outputs = outputs[: , 0 : args.num_classes]
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total

        return loss, acc

    def fine_tune(self, args, dataset, idxs, model):
        trainloader = self.test_split(dataset, list(idxs))
        device = args.device
        criterion = nn.NLLLoss().to(device)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        model.train()
        for i in range(args.ft_round):
            for batch_idx, (images, label_g) in enumerate(trainloader):
                images, labels = images.to(device), label_g.to(device)

                # compute loss
                model.zero_grad()
                log_probs, protos = model(images)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

        return model.state_dict()







def test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos=[]):
 with torch.no_grad():
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    criterion = nn.NLLLoss().to(device)

    acc_list_g = []
    acc_list_l = []
    loss_list = []
    for idx in range(args.num_users):
        total=0
        correct=0


        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)
        c = []
        d = []
        # test (local model)
        model.eval()
        for batch_idx, (images, labels) in enumerate(fda_dataset_read1(domain=idx)):
            # images, labels = images.to(self.device), labels_g.to(self.device)
            images, labels = images.to(device), labels.to(device)
            # print('456465456456456456456456456456',len(labels))
            # print(len(labels))
            model.zero_grad()
            outputs, protos = model(images)
            # print('22222222222222',outputs.shape)
            # print('11111111111111',labels.shape)

            batch_loss = criterion(outputs, labels.long().squeeze())
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            # pred_labels = pred_labels.view(-1)


            # correct += torch.sum(torch.eq(pred_labels, labels.long().squeeze())).item()
            correct += (pred_labels == labels.long().squeeze()).sum().item()

            # print('1111111111111',pred_labels)
            # print('2222222222222',labels.long().squeeze().shape)
            total += labels.size(0)
            if idx == 1:
                # print('22222222222', pred_labels)
                # print('2222222222222', labels.long().squeeze())
                c.extend(pred_labels)
                d.extend(labels.long().squeeze())
                # print('1111111111111111111111111111', pred_labels)
                # print('2222222222222222222222222222', labels.long().squeeze())
                # print()
                # np.save('/home/yq/FedProto-main/biaoqian/zhenshi.npy',d)
                # np.save('/home/yq/FedProto-main/biaoqian/pre.npy',c)
                #
                #
                # torch.save(local_model_list[1],'/home/yq/FedProto-main/model.pt')
                np.save('/home/yq/FedProto-main/bianqian1/11zhenshi.npy', d)
                np.save('/home/yq/FedProto-main/bianqian1/11pre.npy', c)
            # total += len(labels)
            # print('bbbbbbbbbbbbbbbbbb', total)
        # print('aaaaaaaaaaaaaaaaaaa',correct,'bbbbbbbbbbbbbbbbbb',total)
        acc = correct / total
        print('aaaaaaaaaaaaaaaaaaa', correct, 'bbbbbbbbbbbbbbbbbb', total)
        if idx == 1:
            torch.save(local_model_list[1], '/home/yq/FedProto-main/model.pt')
        print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc))
        acc_list_l.append(acc)
        total = 0
        correct = 0
        e=[]
        f=[]
        # test (use global proto)
        if global_protos!=[]:
            for batch_idx, (images, labels) in enumerate(fda_dataset_read1(domain=idx)):
                images, labels = images.to(device), labels.to(device)
                # print('ooooooooooooooooooo',labels.shape)
                model.zero_grad()
                outputs, protos = model(images)

                # compute the dist between protos and global_protos
                a_large_num = 100
                dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device)  # initialize a distance matrix
                classes_list= np.array([0,1,2,3,4,5,6])
                for i in range(images.shape[0]):
                    for j in range(args.num_classes):
                        if j in global_protos.keys() and j in classes_list:
                            d = loss_mse(protos[i, :], global_protos[j][0])
                            dist[i, j] = d

                # prediction
                _, pred_labels = torch.min(dist, 1)
                # print('nininininininini',pred_labels.shape)
                # pred_labels = pred_labels.view(-1)
                # print('asdasdasdasdasdasd',pred_labels.shape)
                # correct += torch.sum(torch.eq(pred_labels, labels.long().squeeze())).item()
                # total += len(labels)
                correct += (pred_labels == labels.long().squeeze()).sum().item()

                # print('1111111111111',pred_labels)
                # print('2222222222222',labels.long().squeeze().shape)
                total += labels.size(0)
                # print('111111111',pred_labels)
                # print('222222222',labels.long().squeeze())
                if idx == 1:
                    # print('333333333', pred_labels)
                    # print('444444444', labels.long().squeeze())
                    e.extend(labels.long().squeeze())
                    f.extend(pred_labels)
                    # print('aaaaaaaaaaaaaaaaaaaaaaaaaaaa',pred_labels)
                    # print('bbbbbbbbbbbbbbbbbbbbbbbbbbbb',labels.long().squeeze())
                    np.save('/home/yq/FedProto-main/bianqian1/111zhenshi123.npy', e)
                    np.save('/home/yq/FedProto-main/bianqian1/111pre123.npy', f)
                # compute loss
                proto_new = copy.deepcopy(protos.data)
                i = 0
                for label in labels:
                    if label.item() in global_protos.keys():
                        proto_new[i, :] = global_protos[label.item()][0].data
                    i += 1
                loss2 = loss_mse(proto_new, protos)
                if args.device == 'cuda':
                    loss2 = loss2.cpu().detach().numpy()
                else:
                    loss2 = loss2.detach().numpy()

            acc = correct / total
            print('aaaaaaaaaaaaaaaaaaa', correct, 'bbbbbbbbbbbbbbbbbb', total)
            # print('fffffffffffffffffffffffffffff',acc)
            if idx ==1:


                 torch.save(local_model_list[1], '/home/yq/FedProto-main/111model123.pt')

                # np.save('/home/yq/FedProto-main/biaoqian/zhenshi123.npy', e)
                # np.save('/home/yq/FedProto-main/biaoqian/pre123.npy', f)
                #
                # torch.save(local_model_list[1], '/home/yq/FedProto-main/model123.pt')

            print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc))
            acc_list_g.append(acc)
            loss_list.append(loss2)

    return acc_list_l, acc_list_g, loss_list


def save_protos(args, local_model_list, test_dataset, user_groups_gt):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)

    agg_protos_label = {}
    for idx in range(args.num_users):
        agg_protos_label[idx] = {}
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)

        model.eval()
        for batch_idx, (images, labels) in enumerate(fda_dataset_read1(domain=idx)):
            images, labels = images.to(device), labels.to(device)

            model.zero_grad()
            outputs, protos = model(images)

            batch_loss = criterion(outputs, labels.long().squeeze())
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            for i in range(len(labels)):
                if labels[i].item() in agg_protos_label[idx]:
                    agg_protos_label[idx][labels[i].item()].append(protos[i, :])
                else:
                    agg_protos_label[idx][labels[i].item()] = [protos[i, :]]

    x = []
    y = []
    d = []
    for i in range(args.num_users):
        for label in agg_protos_label[i].keys():
            for proto in agg_protos_label[i][label]:
                if args.device == 'cuda':
                    tmp = proto.cpu().detach().numpy()
                else:
                    tmp = proto.detach().numpy()
                x.append(tmp)
                y.append(label)
                d.append(i)

    x = np.array(x)
    y = np.array(y)
    d = np.array(d)
    np.save('./' + args.alg + '_protos.npy', x)
    np.save('./' + args.alg + '_labels.npy', y)
    np.save('./' + args.alg + '_idx.npy', d)

    print("Save protos and labels successfully.")

def test_inference_new_het_cifar(args, local_model_list, test_dataset, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    cnt = 0
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            probs, protos = model(images)  # outputs 64*6
            prob_list.append(probs)

        a_large_num = 1000
        outputs = a_large_num * torch.ones(size=(images.shape[0], 100)).to(device)  # outputs 64*10
        for i in range(images.shape[0]):
            for j in range(100):
                if j in global_protos.keys():
                    dist = loss_mse(protos[i,:],global_protos[j][0])
                    outputs[i,j] = dist

        _, pred_labels = torch.topk(outputs, 5)
        for i in range(pred_labels.shape[1]):
            correct += torch.sum(torch.eq(pred_labels[:,i], labels)).item()
        total += len(labels)

        cnt+=1
        if cnt==20:
            break

    acc = correct/total

    return acc