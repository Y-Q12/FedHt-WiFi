import torch.utils.data as data
from PIL import Image
import numpy as np
from scipy.io import loadmat
from os import path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import glob
import random
import os
# import numpy as np
from PIL import Image
from torch.utils.data import Dataset
# import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import torch


class DigitFiveDataset(data.Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        super(DigitFiveDataset, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        if img.shape[0] != 1:
            # transpose to Image type,so that the transform function can be used
            img = Image.fromarray(np.uint8(np.asarray(img.transpose((1, 2, 0)))))

        elif img.shape[0] == 1:
            im = np.uint8(np.asarray(img))
            # turn the raw image into 3 channels
            im = np.vstack([im, im, im]).transpose((1, 2, 0))
            img = Image.fromarray(im)

        # do transform with PIL
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return self.data.shape[0]


def load_mnist(base_path):
    mnist_data = loadmat(path.join(base_path, "dataset", "DigitFive", "mnist_data.mat"))
    mnist_train = np.reshape(mnist_data['train_32'], (55000, 32, 32, 1))
    mnist_test = np.reshape(mnist_data['test_32'], (10000, 32, 32, 1))
    # turn to the 3 channel image with C*H*W
    mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
    mnist_train = mnist_train.transpose(0, 3, 1, 2).astype(np.float32)
    mnist_test = mnist_test.transpose(0, 3, 1, 2).astype(np.float32)
    # get labels
    mnist_labels_train = mnist_data['label_train']
    mnist_labels_test = mnist_data['label_test']
    # random sample 25000 from train dataset and random sample 9000 from test dataset
    train_label = np.argmax(mnist_labels_train, axis=1)
    inds = np.random.permutation(mnist_train.shape[0])
    mnist_train = mnist_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnist_labels_test, axis=1)

    mnist_train = mnist_train[:25000]
    train_label = train_label[:25000]
    mnist_test = mnist_test[:9000]
    test_label = test_label[:9000]
    return mnist_train, train_label, mnist_test, test_label


def load_mnist_m(base_path):
    mnistm_data = loadmat(path.join(base_path, "dataset", "DigitFive", "mnistm_with_label.mat"))
    mnistm_train = mnistm_data['train']
    mnistm_test = mnistm_data['test']
    mnistm_train = mnistm_train.transpose(0, 3, 1, 2).astype(np.float32)
    mnistm_test = mnistm_test.transpose(0, 3, 1, 2).astype(np.float32)
    # get labels
    mnistm_labels_train = mnistm_data['label_train']
    mnistm_labels_test = mnistm_data['label_test']
    # random sample 25000 from train dataset and random sample 9000 from test dataset
    train_label = np.argmax(mnistm_labels_train, axis=1)
    inds = np.random.permutation(mnistm_train.shape[0])
    mnistm_train = mnistm_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnistm_labels_test, axis=1)
    mnistm_train = mnistm_train[:25000]
    train_label = train_label[:25000]
    mnistm_test = mnistm_test[:9000]
    test_label = test_label[:9000]
    return mnistm_train, train_label, mnistm_test, test_label


def load_svhn(base_path):
    svhn_train_data = loadmat(path.join(base_path, "dataset", "DigitFive", "svhn_train_32x32.mat"))
    svhn_test_data = loadmat(path.join(base_path, "dataset", "DigitFive", "svhn_test_32x32.mat"))
    svhn_train = svhn_train_data['X']
    svhn_train = svhn_train.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_test = svhn_test_data['X']
    svhn_test = svhn_test.transpose(3, 2, 0, 1).astype(np.float32)
    train_label = svhn_train_data["y"].reshape(-1)
    test_label = svhn_test_data["y"].reshape(-1)
    inds = np.random.permutation(svhn_train.shape[0])
    svhn_train = svhn_train[inds]
    train_label = train_label[inds]
    svhn_train = svhn_train[:25000]
    train_label = train_label[:25000]
    svhn_test = svhn_test[:9000]
    test_label = test_label[:9000]
    train_label[train_label == 10] = 0
    test_label[test_label == 10] = 0
    return svhn_train, train_label, svhn_test, test_label


def load_syn(base_path):
    print("load syn train")
    syn_train_data = loadmat(path.join(base_path, "dataset", "DigitFive", "synth_train_32x32.mat"))
    print("load syn test")
    syn_test_data = loadmat(path.join(base_path, "dataset", "DigitFive", "synth_test_32x32.mat"))
    syn_train = syn_train_data["X"]
    syn_test = syn_test_data["X"]
    syn_train = syn_train.transpose(3, 2, 0, 1).astype(np.float32)
    syn_test = syn_test.transpose(3, 2, 0, 1).astype(np.float32)
    train_label = syn_train_data["y"].reshape(-1)
    test_label = syn_test_data["y"].reshape(-1)
    syn_train = syn_train[:25000]
    syn_test = syn_test[:9000]
    train_label = train_label[:25000]
    test_label = test_label[:9000]
    train_label[train_label == 10] = 0
    test_label[test_label == 10] = 0
    return syn_train, train_label, syn_test, test_label


def load_usps(base_path):
    usps_dataset = loadmat(path.join(base_path, "dataset", "DigitFive", "usps_28x28.mat"))
    usps_dataset = usps_dataset["dataset"]
    usps_train = usps_dataset[0][0]
    train_label = usps_dataset[0][1]
    train_label = train_label.reshape(-1)
    train_label[train_label == 10] = 0
    usps_test = usps_dataset[1][0]
    test_label = usps_dataset[1][1]
    test_label = test_label.reshape(-1)
    test_label[test_label == 10] = 0
    usps_train = usps_train * 255
    usps_test = usps_test * 255
    usps_train = np.concatenate([usps_train, usps_train, usps_train], 1)
    usps_train = np.tile(usps_train, (4, 1, 1, 1))
    train_label = np.tile(train_label,4)
    usps_train = usps_train[:25000]
    train_label = train_label[:25000]
    usps_test = np.concatenate([usps_test, usps_test, usps_test], 1)
    return usps_train, train_label, usps_test, test_label


def digit5_dataset_read11(base_path, domain, batch_size):
    if domain == "mnist":
        train_image, train_label, test_image, test_label = load_mnist(base_path)
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

        return data_A, label_A

    def __len__(self):
        return len(self.files_A)

def digit5_dataset_read2(base_path,domain,batch_size):
    # train_loader_1 = DataLoader(ImageDataset('./datasets_1/', unaligned=True, mode='train'),
    #                           batch_size=120, shuffle=True)
    test_loader = DataLoader(ImageDataset_test(base_path, domain=domain, unaligned=True ),
                               batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(ImageDataset(base_path, domain=domain,unaligned=True ),
                                batch_size=batch_size, shuffle=True)
    return train_loader, test_loader



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

        return self.x_data[index].float(), self.y_data[index].long()

    def __len__(self):
        return self.len


def digit5_dataset_read(data_path, domain_id,batch_size):
    # loading path
    train_dataset = torch.load(os.path.join(data_path, "train_" + domain_id + ".pt"))
    test_dataset = torch.load(os.path.join(data_path, "test_" + domain_id + ".pt"))

    # Loading datasets
    train_dataset = Load_Dataset(train_dataset,)
    test_dataset = Load_Dataset(test_dataset,)

    # Dataloaders
    # batch_size = 5
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True,)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=True, )
    return train_loader, test_loader