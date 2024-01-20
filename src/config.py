import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import MyWorks.adcdata as adcdata

if_bit_slicing = True

## Use global parameters (below) for all layers or layer specific parameters
val = True
ifglobal_weight_bits = val
ifglobal_weight_bit_frac = val
ifglobal_input_bits = val
ifglobal_input_bit_frac = val
ifglobal_xbar_col_size = val
ifglobal_xbar_row_size = val
ifglobal_tile_col = val
ifglobal_tile_row = val
ifglobal_bit_stream = val
ifglobal_bit_slice = val
ifglobal_adc_bit = val
ifglobal_acm_bits = val
ifglobal_acm_bit_frac = val
ifglobal_xbmodel = False
ifglobal_xbmodel_weight_path = False
ifglobal_dataset = True  # if True data collected from all layers

## Fixed point arithmetic configurations
weight_bits = 8
weight_bit_frac = 7
input_bits = 8    #4 
input_bit_frac = 5  #3

## Tiling configurations
tile_row = 8
tile_col = 8
xbar_row_size = 64       #64
xbar_col_size = 64

## Bit-slicing configurations
bit_stream = 1  # 1
bit_slice = 1   # 2
adc_bit = 7
acm_bits = 32
acm_bit_frac = 24

## Real ADC Characteristics:
is_adc = True
adc_id = 0
wnoise_gamma = 0
adc_dict = {
    0:'ideal',
    1:'ss_lin',
    2:'cco_lin'
}
adc_th = adcdata.__dict__[adc_dict[adc_id]](adc_bits=adc_bit, adc_f_bits=0, adc_index=0)

print('MVM Config:--------------')
print('Bit Stream: ', bit_stream)
print('Bit Slice: ', bit_slice)
print('ADC Bit: ', adc_bit)
print('Xbar size: ', xbar_col_size, xbar_row_size)
print('Real ADC Characteristics: ', is_adc, adc_dict[adc_id])
print('Xbar Weight Std Gamma', wnoise_gamma)

## GENIEx configurations
loop = False # executes GENIEx with batching when set to False

## GENIEx data collection configuations
dataset = False
direc = 'geniex_dataset'  # folder containing geneix dataset
rows = 1 # num of crossbars in row dimension
cols = 1 # num of crossbars in col dimension
Gon = 1/100
Goff = 1/600
Vmax =0.25

# creating directory for dataset collection
if dataset:
    if not os.path.exists(direc):
        os.mkdir(direc)    

non_ideality = False
class NN_model(nn.Module):
    def __init__(self, N):
         super(NN_model, self).__init__()
         print ("WARNING: crossbar sizes with different row annd column dimension not supported.")
         self.fc1 = nn.Linear(N**2+N, 500)
         # self.bn1 = nn.BatchNorm1d(500)
         self.relu1 = nn.ReLU(inplace=True)
         self.do2 = nn.Dropout(0.5)
         self.fc3 = nn.Linear(500,N)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu1(out)
        # out = self.do2(out)
        out = self.fc3(out)
        return out

xbmodel = NN_model(xbar_row_size)
#xbmodel = None
xbmodel_weight_path = '/home/nano01/a/snegi/Projects/NAS/resnet_cutout_ni/dataset_xb/final_xb_models/XB_128_stream1slice207dropout50epochs.pth.tar' #XB_'+str(xb_size)+'_stream1slice2.pth.tar'
#xbmodel_weight_path = None

# Dump the current global configurations
def dump_config():
    param_dict = {'weight_bits':weight_bits, 'weight_bit_frac':weight_bit_frac, 'input_bits':input_bits, 'input_bit_frac':input_bit_frac, 
                  'xbar_row_size':xbar_row_size, 'xbar_col_size':xbar_col_size, 'tile_row':tile_row, 'tile_col':tile_col,
                  'bit_stream':bit_stream, 'bit_slice':bit_slice, 'adc_bit':adc_bit, 'acm_bits':acm_bits, 'acm_bit_frac':acm_bit_frac,
                  'non-ideality':non_ideality, 'xbmodel':xbmodel, 'xbmodel_weight_path':xbmodel_weight_path}

    print("==> Functional simulator configurations:", end=' ')
    for key, val in param_dict.items():
        t_str = key + '=' + str(val)
        print (t_str, end=', ')
    print('\n')
