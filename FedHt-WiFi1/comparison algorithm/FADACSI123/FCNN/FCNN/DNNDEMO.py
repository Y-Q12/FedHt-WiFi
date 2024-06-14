import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
from data import Get_dataloader
from torch.autograd import Function
import time
torch.cuda.set_device(1)


class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

#AC MSEloss 调完学习率，调特征提取参数大小，通道数, 各项参数不再动修改
class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.feature = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(True),
                                      nn.MaxPool2d(kernel_size=4),
                                   nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(True),
                                   nn.MaxPool2d(kernel_size=4),



                                     )
        self.class_d = nn.Sequential(nn.Linear(8000, 256),
                                   nn.ReLU(True),
                                   nn.Linear(256, 7))
        self.domain_d = nn.Sequential(nn.Linear(8000, 256),
                                   nn.ReLU(True),
                                   nn.Linear(256, 2))
        self.GRL = GRL()

    def forward(self, x, alpha):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        class_pre = self.class_d(feature)
        feature1 = GRL.apply(feature, alpha)
        domain_pre = self.domain_d(feature1)
        return class_pre, domain_pre