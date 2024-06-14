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

class ImageDataset(Dataset):

    def __init__(self, root, transforms_=None, unaligned=False,domain='domain' ,mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, 'train_%s' % domain) + '/*.*'))


    def __getitem__(self, index):
        A = np.load(self.files_A[index % len(self.files_A)])


        dat_A = A[:, 1:]
        # print('asd',dat_A.shape)
        domain_A = A[0, :1]
       # label_A = A[0, 1:2]
        label_A = A[0, :1]
        # print('label_A',label_A)  #zyydata hao


        data_A = np.expand_dims(dat_A, 0)
        # print('aaa',data_A.shape)


        return data_A, label_A


    def __len__(self):
        return len(self.files_A)
        # return 1

def Get_dataloader(domain):

    # train_loader_1 = DataLoader(ImageDataset('./datasets_1/', unaligned=True, mode='train'),
    #                           batch_size=120, shuffle=True)
    test_loader_1 = DataLoader(ImageDataset('/home/yq/yqdata2',domain=domain,unaligned=True, mode='test'),
                                  batch_size=5, shuffle=True)
    train_loader_1 = DataLoader(ImageDataset('/home/yq/yqdata2', unaligned=True, mode='train'),
                                  batch_size=30, shuffle=True)
    return test_loader_1, train_loader_1




root_dir='/home/yq/yqdata2/'

class CSLOS_Dataset(Dataset):
    def __init__(self, filelist, domain ):
        self.filelist = filelist
        self.domain = domain
        # self.min_len = 200
        # self.max_len = 2000

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        #
        filename = self.filelist[idx]
        print(filename)
        csi = np.load(os.path.join(root_dir, f"train_{self.domain}", filename))
        data = csi[:, 1:]
        data_A = np.expand_dims(data, 0)
        lable = csi[0, :1]

        return data_A, lable


def digit5_dataset_read(base_path, domain, batch_size):
    if domain == "A":
        train_image, train_label, load_mnist(base_path)
    elif domain == "mnistm":
        train_image, train_label, test_image, test_label = load_mnist_m(base_path)
    elif domain == "svhn":
        train_image, train_label, test_image, test_label = load_svhn(base_path)
    elif domain == "syn":
        train_image, train_label, test_image, test_label = load_syn(base_path)
    elif domain == "usps":
        train_image, train_label, test_image, test_label = load_usps(base_path)
    else:
        raise NotImplementedError("Domain {} Not Implemented".format(domain))
    # define the transform function
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # raise train and test data loader
    train_dataset = DigitFiveDataset(data=train_image, labels=train_label, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataset = DigitFiveDataset(data=test_image, labels=test_label, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader, test_loader
