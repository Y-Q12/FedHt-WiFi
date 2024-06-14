import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, data_parallel=True):
        super(CNN, self).__init__()
        encoder = nn.Sequential()
        encoder.add_module("conv1", nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2))
        encoder.add_module("bn1", nn.BatchNorm2d(16))
        encoder.add_module("relu1", nn.ReLU())
        encoder.add_module("maxpool1", nn.MaxPool2d(kernel_size=4, stride=2, padding=1, ceil_mode=False))
        encoder.add_module("conv2", nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2))
        encoder.add_module("bn2", nn.BatchNorm2d(32))
        encoder.add_module("relu2", nn.ReLU())
        encoder.add_module("maxpool2", nn.MaxPool2d(kernel_size=4, stride=2, padding=1, ceil_mode=False))
        # encoder.add_module("conv3", nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2))
        # encoder.add_module("bn3", nn.BatchNorm2d(64))
        # encoder.add_module("relu3", nn.ReLU())
        # encoder.add_module("maxpool3", nn.MaxPool2d(kernel_size=4, stride=2, padding=1, ceil_mode=False))
        if data_parallel:
            self.encoder = nn.DataParallel(encoder)
        else:
            self.encoder = encoder
        linear = nn.Sequential()
        linear.add_module("fc1", nn.Linear(2048, 2048))
        # linear.add_module("bn4", nn.BatchNorm1d(3072))
        # linear.add_module("relu4", nn.ReLU())
        # linear.add_module("dropout", nn.Dropout())
        # linear.add_module("fc2", nn.Linear(3072, 2048))
        linear.add_module("bn5", nn.BatchNorm1d(2048))
        linear.add_module("relu5", nn.ReLU())
        if data_parallel:
            self.linear = nn.DataParallel(linear)
        else:
            self.linear = linear

    def forward(self, x):
        # print(x.shape)
        x = x.view(-1,1,9,128)
        batch_size = x.size(0)
        feature = self.encoder(x)
        # print(feature.shape)
        feature = feature.view(batch_size, 2048)
        feature = self.linear(feature)
        # print(feature.shape)
        return feature


class Classifier(nn.Module):
    def __init__(self, data_parallel=True):
        super(Classifier, self).__init__()
        linear = nn.Sequential()
        linear.add_module("fc", nn.Linear(2048, 7))
        if data_parallel:
            self.linear = nn.DataParallel(linear)
        else:
            self.linear = linear

    def forward(self, x):
        x = self.linear(x)
        # print(x.shape)
        return x

