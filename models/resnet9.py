import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import time
__all__ = ['net']

class resnet(nn.Module):

    def __init__(self):
        super(resnet, self).__init__()

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu2(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.mp2(out)

        residual1 = out.clone()

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)

        out = out + residual1

        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu5(out)
        out = self.mp5(out)

        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu6(out)
        out = self.mp6(out)

        residual2 = out.clone()

        out = self.conv7(out)
        out = self.bn7(out)
        out = self.relu7(out)
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.relu8(out)
        
        out = out + residual2

        out = self.adaptivemp(out)
        out = self.flat(out)
        out = self.dropout(out)
        out = self.fc(out)

        return out

class ResNet9(resnet):

    def __init__(self, num_classes=10):
        super(ResNet9, self).__init__()
        # conv1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        # conv 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.mp2 = nn.MaxPool2d(2)
        # residual 1
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        # conv3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)
        self.mp5 = nn.MaxPool2d(2)
        # conv4
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)
        self.mp6 = nn.MaxPool2d(2)
        # residual 2
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU(inplace=True)
        # classifier
        self.adaptivemp = nn.AdaptiveMaxPool2d((1,1))
        self.flat = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512, 10)

def net(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    num_classes = 100
    return ResNet9(num_classes=num_classes)
    #if dataset == 'cifar100':