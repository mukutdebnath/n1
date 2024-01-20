import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from MyWorks.classes import *
    
class MLP2_SS_ADC(nn.Module):
    def __init__(self, relu_range, adc_bits):
        super(MLP2_SS_ADC,self).__init__()
        self.relu_range = relu_range  
        self.adc_bits = adc_bits
        self.fc1 = nn.Linear(28*28, 100)
        self.SS_ADC =  SSADC(self.relu_range, self.adc_bits)
        self.droput = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100,10)
        
        
    def forward(self,x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = self.SS_ADC(x)
        x = self.droput(x)
        x = self.fc2(x)
        return x
    
class MLP2_CCO_ADC(nn.Module):
    def __init__(self, relu_range, adc_bits):
        super(MLP2_CCO_ADC,self).__init__()
        self.relu_range = relu_range  
        self.adc_bits = adc_bits    
        self.fc1 = nn.Linear(28*28, 100)     
        self.CCO_ADC =  CCOADC(self.relu_range, self.adc_bits)
        self.droput = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100,10)
        
        
    def forward(self,x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = self.CCO_ADC(x)
        x = self.droput(x)
        x = self.fc2(x)
        return x
    
class MLP2_CCO_ADC_MC(nn.Module):
    def __init__(self, relu_range, adc_bits):
        super(MLP2_CCO_ADC_MC,self).__init__()
        self.relu_range = relu_range  
        self.adc_bits = adc_bits    
        self.fc1 = nn.Linear(28*28, 100)     
        self.CCO_ADC_MC =  CCOADC_MC(100, self.relu_range, self.adc_bits)
        self.droput = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100,10)
        
        
    def forward(self,x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = self.CCO_ADC_MC(x)
        x = self.droput(x)
        x = self.fc2(x)
        return x

class MLP2_Ideal_ADC(nn.Module):
    def __init__(self, relu_range, adc_bits):
        super(MLP2_Ideal_ADC,self).__init__()
        self.relu_range = relu_range  
        self.adc_bits = adc_bits    
        self.fc1 = nn.Linear(28*28, 100)     
        self.Ideal_ADC =  IdealADC(self.relu_range, self.adc_bits)
        self.droput = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100,10)
        
        
    def forward(self,x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = self.Ideal_ADC(x)
        x = self.droput(x)
        x = self.fc2(x)
        return x