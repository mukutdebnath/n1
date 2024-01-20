import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import time
import pdb

import src.config as cfg

if cfg.if_bit_slicing and not cfg.dataset:
    from src.pytorch_mvm_class_v3 import *
elif cfg.dataset:
    from geneix.pytorch_mvm_class_dataset import *   # import mvm class from geneix folder
else:
    from src.pytorch_mvm_class_no_bitslice import *
    
class MLP2_mvm(nn.Module):
    def __init__(self):
        super(MLP2_mvm,self).__init__()
        self.fc1 = Linear_mvm(28*28, 100)
        self.droput = nn.Dropout(0.2)
        self.fc2 = Linear_mvm(100,10)
        
    def forward(self,x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = self.droput(x)
        x = self.fc2(x)
        return x