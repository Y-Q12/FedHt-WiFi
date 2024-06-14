import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict


class feat_bootleneck(nn.Module):
    def __init__(self):
        super(feat_bootleneck, self).__init__()
        self.feature = nn.Sequential(
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(True),
                                      )


        self.feature2 = nn.Sequential(nn.Linear(16000,2048),
                                      nn.ReLU(True),
                                     nn.Linear(2048, 1024),
                                     nn.ReLU(True),
                                     nn.Linear(1024, 256),
                                      )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)

        x = self.feature2(x)

        return x

class feat_classifier(nn.Module):
    def __init__(self):
        super(feat_classifier, self).__init__()
        self.class_d = nn.Linear(256, 7)



    def forward(self, x):
        x = self.class_d(x)

        return x



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feature = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(True),
                                     nn.MaxPool2d(kernel_size=2),

                                     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(True),
                                     nn.MaxPool2d(kernel_size=2),

                                     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(True),
                                     nn.MaxPool2d(kernel_size=4)
                                     )


    def forward(self, x):
        x = self.feature(x)


        return x