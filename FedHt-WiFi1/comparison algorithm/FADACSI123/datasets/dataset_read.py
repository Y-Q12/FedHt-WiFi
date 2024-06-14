import sys
import numpy as np
import glob
import random
import os
# import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import torch

sys.path.append('../loader')
from unaligned_data_loader import UnalignedDataLoader,fda_DataLoader
from svhn import load_svhn
from mnist import load_mnist
from mnist_m import load_mnistm
from usps_ import load_usps
from synth_number import load_syn


def return_dataset(data, scale=False, usps=False, all_use='no'):
    if data == 'svhn':
        train_image, train_label, test_image, test_label = load_svhn()
    if data == 'mnist':
        train_image, train_label, test_image, test_label = load_mnist(scale=scale, usps=usps, all_use=all_use)
        print(train_image.shape)
    if data == 'usps':
        train_image, train_label, test_image, test_label = load_usps(all_use=all_use)
    if data == 'mnistm':
        train_image, train_label, test_image, test_label = load_mnistm()
    if data == 'synth':
        train_image, train_label, test_image, test_label = load_syntraffic()
    if data == 'gtsrb':
        train_image, train_label, test_image, test_label = load_gtsrb()
    if data == 'syn':
        train_image, train_label, test_image, test_label = load_syn()

    return train_image, train_label, test_image, test_label

def fda_dataset_read1(domain, batch_size, scale=False, all_use='no'):
    S={}
    S_test={}
    usps = False
    if domain=='usps' or domain =='usps':
        usps = True
    train_data, train_label, test_data, test_label = return_dataset(domain, scale=scale,usps=usps,all_use=all_use)

    S['imgs'] = train_data
    S['labels'] = train_label
    S_test['imgs'] = test_data
    S_test['labels'] = test_label
    scale = 32 if domain == 'synth' else 32 if domain == 'usps' or domain == 'usps' else 32
    train_loader = fda_DataLoader()
    train_loader.initialize(S, batch_size, scale)
    dataset = train_loader.load_data()
    test_loader = fda_DataLoader()
    test_loader.initialize(S_test, batch_size, scale)
    dataset_test = test_loader.load_data()
    return dataset, dataset_test


def dataset_read(source, target, batch_size, scale=False, all_use='no'):
    S = {}
    S_test = {}
    T = {}
    T_test = {}
    usps = False
    if source == 'usps' or target == 'usps':
        usps = True

    domain_all = ['mnistm', 'mnist', 'usps', 'svhn', 'syn']
    domain_all.remove(source)
    train_source, s_label_train, test_source, s_label_test = return_dataset(source, scale=scale,usps=usps, all_use=all_use)

    train_target, t_label_train, test_target, t_label_test = return_dataset(domain_all[0], scale=scale, usps=usps,
                                                                                all_use=all_use)
    for i in range(1, len(domain_all)):
        train_target_, t_label_train_, test_target_, t_label_test_ = return_dataset(domain_all[i], scale=scale, usps=usps, all_use=all_use)
        train_target = np.concatenate((train_target, train_target_), axis=0)
        t_label_train = np.concatenate((t_label_train, t_label_train_), axis=0)
        test_target = np.concatenate((test_target, test_target_), axis=0)
        t_label_test = np.concatenate((t_label_test, t_label_test_), axis=0)

    # print(domain)
    print('Source Training: ', train_source.shape)
    print('Source Training label: ', s_label_train.shape)
    print('Source Test: ', test_source.shape)
    print('Source Test label: ', s_label_test.shape)

    print('Target Training: ', train_target.shape)
    print('Target Training label: ', t_label_train.shape)
    print('Target Test: ', test_target.shape)
    print('Target Test label: ', t_label_test.shape)




    S['imgs'] = train_source
    S['labels'] = s_label_train
    T['imgs'] = train_target
    T['labels'] = t_label_train

    # input target samples for both 
    S_test['imgs'] = test_target
    S_test['labels'] = t_label_test
    T_test['imgs'] = test_target
    T_test['labels'] = t_label_test
    scale = 32 if source == 'synth' else 32 if source == 'usps' or target == 'usps' else 32
    train_loader = UnalignedDataLoader()
    train_loader.initialize(S, T, batch_size, batch_size, scale=scale)
    dataset = train_loader.load_data()
    test_loader = UnalignedDataLoader()
    test_loader.initialize(S_test, T_test, batch_size, batch_size, scale=scale)
    dataset_test = test_loader.load_data()
    return dataset, dataset_test



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

        return {'img': data_A, 'label': label_A}

    def __len__(self):
        return len(self.files_A)
        # return 1
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

        return {'img': data_A, 'label': label_A}

    def __len__(self):
        return len(self.files_A)

def fda_dataset_read(root,domain,batch_size):
    # train_loader_1 = DataLoader(ImageDataset('./datasets_1/', unaligned=True, mode='train'),
    #                           batch_size=120, shuffle=True)
    dataset_test = DataLoader(ImageDataset_test(root, domain=domain, unaligned=True ),
                               batch_size=batch_size, shuffle=True,drop_last=True)
    dataset = DataLoader(ImageDataset(root, domain=domain,unaligned=True ),
                                batch_size=batch_size, shuffle=True,drop_last=True)
    return dataset, dataset_test










class Load_Dataset(Dataset):
    def __init__(self, dataset):
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train)
            y_train = torch.from_numpy(y_train).long()

        if X_train.shape.index(min(X_train.shape[1], X_train.shape[2])) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        self.x_data = X_train
        self.y_data = y_train

        self.num_channels = X_train.shape[1]


        self.transform = None

        self.len = X_train.shape[0]

    def __getitem__(self, index):
        if self.transform is not None:
            output = self.transform(self.x_data[index].view(self.num_channels, -1, 1))
            self.x_data[index] = output.view(self.x_data[index].shape)

        # return self.x_data[index].float(), self.y_data[index].long()
        return {'img': self.x_data[index].float(), 'label': self.y_data[index].long()}

    def __len__(self):
        return self.len


def data_generator(data_path, domain_id):
    # loading path
    train_dataset = torch.load(os.path.join(data_path, "train_" + domain_id + ".pt"))
    test_dataset = torch.load(os.path.join(data_path, "test_" + domain_id + ".pt"))

    # Loading datasets
    train_dataset = Load_Dataset(train_dataset,)
    test_dataset = Load_Dataset(test_dataset,)

    # Dataloaders
    batch_size = 5
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=20,
                                               shuffle=True,)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=5,
                                              shuffle=True, )
    return train_loader, test_loader