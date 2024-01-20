import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
my_dir = os.path.join(root_dir, "MyWorks")

sys.path.insert(0, my_dir) 

import torch
import torch.nn as nn

from MyWorks.adcInterface import ReLU_adc

class LeNet5_adc(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5_adc, self).__init__()        
        self.conv1=nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.bn1=nn.BatchNorm2d(6)
        self.relu1=ReLU_adc()
        self.mp1=nn.MaxPool2d(kernel_size = 2, stride = 2)        
        self.conv2=nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.bn2=nn.BatchNorm2d(16)
        self.relu2=ReLU_adc()
        self.mp2=nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(400, 120)
        self.relu3 = ReLU_adc()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = ReLU_adc()
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out_relu1 = out
        out = self.mp1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out_relu2 = out
        out = self.mp2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu3(out)
        out_relu3 = out
        out = self.fc2(out)
        out = self.relu4(out)
        out_relu4 = out
        out = self.fc3(out)
        return out, out_relu1, out_relu2, out_relu3, out_relu4