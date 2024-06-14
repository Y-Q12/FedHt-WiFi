import numpy as np
from PIL import Image
from torchvision import  transforms
import torch
import random
# a = np.load('/home/yq/zyydata/test_A/walk_all_173.npy')
# print(a)
# b = np.load('/home/yq/yqdata/test_A/wave_jianfei_85.npy')
# print(b)
# c = np.load('/home/yq/yqdata2/test_A/wave_zhifeng_96.npy')
# print(c.shape)
# d = np.load('/home/yq/zsx36/test_A/Z_cf_49.npy')
# print(d)
# source_domain = ['C','D','E']
# df =len(source_domain)
# print(df)
# def step_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  #为GPU设置随机种子
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#
#
# print(step_seed(12))

x_train=np.load('/home/yq/trainX.npy')
x_train=x_train.reshape(-1,100,8)
print(x_train.shape)
# data=addGaussianNoise(x_train,20)
img_pil = Image.fromarray(x_train)
transform=transforms.RandomHorizontalFlip(p=1)
transform(x_train)