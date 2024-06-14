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


def step_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


step_seed(12)


class ImageDataset(Dataset):

    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, '%s_A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s_B' % mode) + '/*.*'))
        self.files_C = sorted(glob.glob(os.path.join(root, '%s_C' % mode) + '/*.*'))
   
        # self.files_G = sorted(glob.glob(os.path.join(root, '%s_G' % mode) + '/*.*'))


    def __getitem__(self, index):
        A = np.load(self.files_A[index % len(self.files_A)])
        B = np.load(self.files_B[(index) % len(self.files_B)])
        C = np.load(self.files_C[(index) % len(self.files_C)])
      


        dat_A = A[:, 1:]
        # print('asd',dat_A.shape)
        domain_A = A[0, :1]
       # label_A = A[0, 1:2]
        label_A = A[0, :1]
        # print('label_A',label_A)  #zyydata hao
        dat_B = B[:, 1:]
        domain_B = B[0, :1]
        #label_B = B[0, 1:2]
        label_B = B[0, :1]
        #fake_B
        # dat_B = B[:, 1:]
        # domain_B = B[0, :1]
        # label_B = B[0, :1]

        dat_C = C[:, 1:]
        domain_C = C[0, :1]
        label_C = C[0,: 1]
    

        data_A = np.expand_dims(dat_A, 0)
        # print('aaa',data_A.shape)
        data_B = np.expand_dims(dat_B, 0)
        data_C = np.expand_dims(dat_C, 0)
   

        return {'A': data_A, 'label_A': label_A, 'domain_A': domain_A, 'B': data_B, 'label_B': label_B, 'domain_B':domain_B,
                'C': data_C, 'label_C': label_C, 'domain_C': domain_C}


    def __len__(self):
        return len(self.files_A)
        # return 1

def Get_dataloader():

    # train_loader_1 = DataLoader(ImageDataset('./datasets_1/', unaligned=True, mode='train'),
    #                           batch_size=120, shuffle=True)
    test_loader_1 = DataLoader(ImageDataset('/root/tf-logs/iclr_2020_code_federated_adversarial_domain_adaptation/data/zyydata', unaligned=True, mode='test'),
                                  batch_size=5, shuffle=True)
    train_loader_1 = DataLoader(ImageDataset('/root/tf-logs/iclr_2020_code_federated_adversarial_domain_adaptation/data/zyydata', unaligned=True, mode='train'),
                                  batch_size=30, shuffle=True)

    #val_loader = DataLoader(ImageDataset('./datasets_1/', unaligned=True, mode='test'),
    #                         batch_size=1, shuffle=True)

    #km_loader = DataLoader(ImageDataset('./datasets_1/', unaligned=True, mode='train'),
    #                        batch_size=1, shuffle=False)
    # return train_loader, test_loader, GAN_train_loader, train_loader_1, test_loader_1, GAN_train_loader_1
    return test_loader_1 ,train_loader_1

