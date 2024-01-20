import os
os.environ['CUDA VISIBLE DEVICES'] = '0,1'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.nn import init
import math
import numpy as np
from src.mvm_v3 import *
import pdb
import time
torch.set_printoptions(threshold=10000)

# Custom conv2d formvm function: Doesn't work for back-propagation
pretrained_model_path = 'final_64x64_mlp2layer_xbar_64x64_100_all_v2_dataset_500_100k_standard_sgd.pth.tar'
pretrained_model = torch.load(pretrained_model_path)


# # pretrained_model = torch.load('final_64x64_mlp2layer_xbar_64x64_100_all_new_standard_sgd.pth.tar')

class NN_model(nn.Module):
    def __init__(self):
         super(NN_model, self).__init__()
         # N=64
         self.fc1 = nn.Linear(4160, 500)
         # self.bn1 = nn.BatchNorm1d(500)
         self.relu1 = nn.ReLU(inplace=True)
         self.do2 = nn.Dropout(0.5)
         self.fc3 = nn.Linear(500,64)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu1(out)
        # out = self.do2(out)
        out = self.fc3(out)
        return out
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = NN_model()
#model.cuda() 
#model.eval()
#model.load_state_dict(pretrained_model['state_dict'])
#model = torch.nn.DataParallel(model) 

class Conv2d_mvm_function(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # +--------------------------+
    # |            MVM           |   
    # +--------------------------+
    def forward(ctx, flatten_binary_input, flatten_input_sign_temp, flatten_input_sign_xbar, output, bias_addr, weight_row, weight_col, output_row, output_col, input_pad, input_batch, pos, neg, xbars, zero_mvmtensor, shift_add_bit_stream,
                shift_add_bit_slice, output_reg, output_analog, Goffmat, G_real_flatten0, G_real_flatten1, G_real0, G_real1, model, tile_row, tile_col, bias=None, stride=1,
                padding=0, dilation=1, groups=1, bit_slice=2, bit_stream=1, weight_bits=16, weight_bit_frac=-1, input_bits=16, input_bit_frac=-1, adc_bit=-1, 
                acm_bits=16, acm_bit_frac=-1, ind=False, loop = True):

        bit_slice_num = weight_bits//bit_slice
        bit_stream_num = input_bits//bit_stream
        num_pixel = tile_row*tile_col
        unfold = nn.Unfold(kernel_size=(weight_row, weight_row), stride=(stride[0], stride[1]))       
        input_patch_row = (tile_row-1)*stride[0] + weight_row
        stride_input_row = stride[0]*tile_row
        input_patch_col = (tile_col-1)*stride[1] + weight_col
        stride_input_col = stride[1]*tile_col
        
        #Tile size should be a multiple of output feature map size
        assert output_row%tile_row == 0 and output_col%tile_col == 0, "Tile size should be a multiple of output feature map size"
#        print('output:{}'.format(output.shape))
        for i in range(math.ceil(output_row/tile_row)):
            for j in range(math.ceil(output_col/tile_col)):
#                print('{},{}'.format(i,j))
#                pdb.set_trace()
                input_temp = unfold(input_pad[:,:, stride_input_row*i:stride_input_row*i+input_patch_row, stride_input_col*j:stride_input_col*j+input_patch_col]).permute(2,0,1) # #patches, batchsize, k^2*I
                input_temp = input_temp.reshape(input_batch*num_pixel,-1)          #new_batch_size = batch_size*#_of_output_pixel    
#                print('shape:{}'.format(input_temp.shape))
                if bit_stream >1:
                    flatten_input_sign = torch.where(input_temp > 0, pos, neg).expand(bit_stream_num,-1,-1).permute(1, 2, 0) 
                    flatten_input_sign_temp[:,:flatten_input_sign.shape[1]] = flatten_input_sign
                    flatten_input_sign_xbar = flatten_input_sign_temp.reshape(input_batch*num_pixel, xbars.shape[1],XBAR_ROW_SIZE, bit_stream_num)
                    input_temp.abs_()

                flatten_binary_input_temp = float_to_16bits_tensor_fast(input_temp, input_bit_frac, bit_stream, bit_stream_num, input_bits)   # batch x n x 16
#                print(flatten_binary_input_temp)
                flatten_binary_input[:,:flatten_binary_input_temp.shape[1]] = flatten_binary_input_temp
                
#                flatten_binary_input[:,:flatten_input_sign.shape[1]] = float_to_16bits_tensor_fast(input_temp, input_bit_frac, bit_stream, bit_stream_num, input_bits, device)   # batch x n x 16
                flatten_binary_input_xbar = flatten_binary_input.reshape((input_batch*num_pixel, xbars.shape[1],XBAR_ROW_SIZE, bit_stream_num))  
#                pdb.set_trace()
                if ind == True:
                    # t1 = time.time()
                    xbars_out = mvm_tensor_ind(zero_mvmtensor, shift_add_bit_stream, shift_add_bit_slice, output_reg, output_analog, Goffmat, G_real_flatten0, G_real0, 
                                               model, loop, flatten_binary_input_xbar, flatten_input_sign_xbar, bias_addr, xbars[0], bit_slice, bit_stream, weight_bits, 
                                               weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bits, acm_bit_frac) - \
                                mvm_tensor_ind(zero_mvmtensor, shift_add_bit_stream, shift_add_bit_slice, output_reg, output_analog, Goffmat, G_real_flatten1, G_real1, 
                                               model, loop, flatten_binary_input_xbar, flatten_input_sign_xbar, bias_addr, xbars[1], bit_slice, bit_stream, weight_bits, 
                                               weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bits, acm_bit_frac) 
                    # t2 = time.time()
                    # print('Time taken: ', t2-t1)
                else:
    #                pdb.set_trace()
                    xbars_out = mvm_tensor(zero_mvmtensor, shift_add_bit_stream, shift_add_bit_slice, output_reg, flatten_binary_input_xbar, flatten_input_sign_xbar, 
                                           bias_addr, xbars[0], bit_slice, bit_stream, weight_bits, weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bits, 
                                           acm_bit_frac, device) - \
                                mvm_tensor(zero_mvmtensor, shift_add_bit_stream, shift_add_bit_slice, output_reg, flatten_binary_input_xbar, flatten_input_sign_xbar,
                                           bias_addr, xbars[1], bit_slice, bit_stream, weight_bits, weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bits, 
                                           acm_bit_frac, device)
                                
    #                print(xbars_out.shape)
                
#                out = xbars_out.reshape(num_pixel, input_batch, -1)
#                out = out.reshape(tile_row, tile_col, input_batch, -1)
#                out = out.permute(2, 3, 0, 1)  
#                output[:,:,i*tile_row:(i+1)*tile_row,j*tile_col:(j+1)*tile_col] = out
                output[:,:,i*tile_row:(i+1)*tile_row,j*tile_col:(j+1)*tile_col] = xbars_out.reshape(tile_row, tile_col, input_batch, -1).permute(2,3,0,1)  ## #batchsize, # o/p channels, tile_row, tile_col 
                
#        print(output)
#        pdb.set_trace()
                #xbars_out.reshape(input_batch, tile_row, tile_col, weight_channels_out).permute(0,3,1,2)
        #        output[:,:,i,j] += xbars_out[:, :weight_channels_out]
                
                
        pdb.set_trace()     
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding 
        ctx.dilation = dilation
        ctx.groups = groups

        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        
        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None
        
        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups) 
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)
            
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None 


class _ConvNd_mvm(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode', 'bit_slice', 'bit_stream','weight_bits', 'weight_bit_frac','input_bits', 'input_bit_frac',
                     'adc_bit','acm_bits', 'acm_bit_frac', 'ind']

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias, padding_mode, bit_slice,
                 bit_stream, weight_bits, weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bits, acm_bit_frac, ind, loop, tile_row, tile_col, check_grad=False):
        super(_ConvNd_mvm, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.bit_slice = bit_slice
        self.bit_stream = bit_stream
        self.weight_bits = weight_bits
        self.weight_bit_frac = weight_bit_frac
        self.input_bits = input_bits
        self.input_bit_frac = input_bit_frac
        self.adc_bit = adc_bit
        self.acm_bits = acm_bits
        self.acm_bit_frac = acm_bit_frac
        self.ind = ind
        self.loop = loop
        self.tile_row = tile_row
        self.tile_col = tile_col
        if check_grad:
            tensor_constructor = torch.DoubleTensor # double precision required to check grad
        else:
            tensor_constructor = torch.Tensor # In PyTorch torch.Tensor is alias torch.FloatTensor

        if transposed:
            self.weight = nn.Parameter(tensor_constructor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = nn.Parameter(tensor_constructor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(tensor_constructor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.model = NN_model()
        self.model.eval()
        self.model.load_state_dict(pretrained_model['state_dict'])
        ## fixed-16: 
        ## sign     : 1 
        ## integer  : 3
        ## fraction : 12


    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

class Conv2d_mvm(_ConvNd_mvm):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', check_grad=False, bit_slice=2, bit_stream=1,
                 weight_bits=16, weight_bit_frac=-1, input_bits=16, input_bit_frac=-1, adc_bit=-1, acm_bits=16, acm_bit_frac=-1, ind=False, loop = True, tile_row=8,
                 tile_col=8):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d_mvm, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, padding_mode, bit_slice, bit_stream, weight_bits,
            weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bits, acm_bit_frac, ind, loop, tile_row, tile_col)
    #@weak_script_method
    def forward(self, input):
        
        self.num_pixel = self.tile_row*self.tile_col
        if self.weight_bit_frac == -1:
            self.weight_bit_frac = self.weight_bits//4*3
        if self.input_bit_frac == -1:
            self.input_bit_frac = self.input_bits//4*3
        if self.acm_bit_frac == -1:
            self.acm_bit_frac = self.acm_bits//4*3
        if self.adc_bit == -1:
            self.adc_bit = int(math.log2(XBAR_ROW_SIZE))
            if self.bit_stream != 1:
                self.adc_bit += self.bit_stream
            if self.bit_slice != 1:
                self.adc_bit += self.bit_slice

#        pdb.set_trace()
        self.weight_channels_out = self.weight.shape[0]
        self.weight_channels_in = self.weight.shape[1]
        self.weight_row = self.weight.shape[2]
        self.weight_col = self.weight.shape[3]

        # weight_pos = torch.where
        self.length = self.weight_channels_in * self.weight_row * self.weight_col
        self.register_buffer('flatten_weight', torch.zeros(2, self.weight_channels_out, self.length))     ## W+ / W-
#        self.register_buffer('flatten_weight', torch.zeros(2, weight_channels_out, length))
        weight = self.weight.reshape((self.weight_channels_out, self.length))
        self.flatten_weight[0] = torch.clamp(weight, min=0)  ## flatten weights
        self.flatten_weight[1] = torch.clamp(weight, max=0).abs()
        self.register_buffer('pos_bit_slice_weight', bit_slicing(self.flatten_weight[0], self.weight_bit_frac, self.bit_slice, self.weight_bits)) ## v2: flatten weights --> fixed point --> bit slice -- v1
        self.register_buffer('neg_bit_slice_weight', bit_slicing(self.flatten_weight[1], self.weight_bit_frac, self.bit_slice, self.weight_bits)) 

#        print(flatten_bit_slice_weight)
        # bitsliced weight into 128x128 xbars 
        # xbar_row separates inputs --> results in a same column with different rows will be added later
        self.xbar_row = math.ceil(self.pos_bit_slice_weight.shape[0]/XBAR_ROW_SIZE)
        self.xbar_col = math.ceil(self.pos_bit_slice_weight.shape[1]/XBAR_COL_SIZE)

        self.register_buffer('weight_xbar', torch.zeros((2,self.xbar_row*XBAR_ROW_SIZE, self.xbar_col*XBAR_COL_SIZE)))
        self.weight_xbar[0,:self.pos_bit_slice_weight.shape[0], :self.pos_bit_slice_weight.shape[1]] = self.pos_bit_slice_weight
        self.weight_xbar[1,:self.neg_bit_slice_weight.shape[0], :self.neg_bit_slice_weight.shape[1]] = self.neg_bit_slice_weight


        self.bit_slice_num = self.weight_bits//self.bit_slice
        self.bit_stream_num = self.input_bits//self.bit_stream
        self.bias_addr = [self.weight_channels_out//int(XBAR_COL_SIZE/self.bit_slice_num), self.weight_channels_out%int(XBAR_COL_SIZE/self.bit_slice_num)]      #####

        self.xbars = self.weight_xbar.unfold(1,XBAR_ROW_SIZE, XBAR_COL_SIZE).unfold(2, XBAR_ROW_SIZE, XBAR_COL_SIZE)
        
        self.input = input
        self.input_batch = self.input.shape[0]
        self.input_channels = self.input.shape[1]     # weight_channels_in == input_channels
        self.input_row = self.input.shape[2] + self.padding[0]*2
        self.input_col = self.input.shape[3] + self.padding[1]*2
        self.register_buffer('input_pad', torch.zeros((self.input_batch, self.input_channels, self.input_row, self.input_col)))
        self.input_pad[:,:,self.padding[0]:self.input_row-self.padding[0],self.padding[1]:self.input_col-self.padding[1]] = self.input
#        print('input device:',input_pad.get_device())
#        pos = torch.ones(input_batch, input_channels, weight_row, weight_col).reshape(input_batch,-1).to(device)
#        neg = pos.clone().fill_(0)
        self.register_buffer('pos', torch.ones(self.input_batch*self.num_pixel, self.input_channels*self.weight_row*self.weight_col))
        self.neg = self.pos.clone().fill_(0)
        
        self.output_row = (self.input_row - self.weight_row)//self.stride[0] + 1
        self.output_col = (self.input_col - self.weight_col)//self.stride[1] + 1 
        self.register_buffer('output', torch.zeros((self.input_batch, self.weight_channels_out, self.output_row, self.output_col)))

        self.register_buffer('flatten_binary_input', torch.zeros(self.input_batch*self.num_pixel, self.xbars.shape[1]*XBAR_ROW_SIZE, self.bit_stream_num))
        
        ## delete unused tensors of weight and inputs
#        del flatten_weight, pos_bit_slice_weight, neg_bit_slice_weight, weight_xbar

        self.register_buffer('flatten_input_sign_temp', torch.zeros(self.input_batch*self.num_pixel, self.xbars.shape[1]*XBAR_ROW_SIZE, self.bit_stream_num))
        self.register_buffer('flatten_input_sign_xbar', torch.zeros(self.input_batch*self.num_pixel, self.xbars.shape[1],XBAR_ROW_SIZE, self.bit_stream_num))
        
        #variables transferred to GPU
        xbars_row = self.xbars.shape[1]  # dimension 0 is for sign 
        xbars_col = self.xbars.shape[2]
        
        self.register_buffer('zero_mvmtensor', torch.zeros(self.input_batch*self.num_pixel, self.xbars.shape[1],XBAR_ROW_SIZE, self.bit_stream_num))
#        shift_add_bit_stream = torch.zeros(bit_stream_num) # input bits = 16
#        for i in range(bit_stream_num):
#            shift_add_bit_stream[i] = 2**(bit_stream*i)
#    
#        shift_add_bit_slice = torch.zeros(bit_slice_num) # 16bit / 2bit-slice
#        for i in range(bit_slice_num):
#            shift_add_bit_slice[-i-1] = 2**(bit_slice*i)        
        self.register_buffer('shift_add_bit_stream', torch.pow(torch.ones(self.bit_stream_num)*2, self.bit_stream*torch.arange(0,self.bit_stream_num)))
        self.register_buffer('shift_add_bit_slice', torch.pow(torch.ones(self.bit_slice_num)*2,  self.bit_slice*torch.arange(self.bit_slice_num, 0, -1))) 
#        pdb.set_trace()
        Gon = 1/100
        Goff = 1/600
        Nstates_slice = 2**self.bit_slice-1        
        if self.bit_stream ==1:
            self.shift_add_bit_stream[-1] *= -1        # last bit --> subtract
            self.shift_add_bit_stream= self.shift_add_bit_stream.expand((self.input_batch*self.num_pixel, xbars_row, xbars_col, 
                                                                                      XBAR_COL_SIZE//self.bit_slice_num, self.bit_stream_num)).transpose(3,4)
            
            self.shift_add_bit_slice=self.shift_add_bit_slice.expand((self.input_batch*self.num_pixel, xbars_row, xbars_col, 
                                                                                    XBAR_COL_SIZE//self.bit_slice_num, self.bit_slice_num))
            self.register_buffer('output_reg', torch.zeros(self.input_batch*self.num_pixel, xbars_row, xbars_col, self.bit_stream_num, XBAR_COL_SIZE//self.bit_slice_num)) # for 32-fixed  
            if self.ind == True:
                self.register_buffer('output_analog', torch.zeros(self.input_batch*self.num_pixel, xbars_row, xbars_col, XBAR_COL_SIZE))
                self.register_buffer('Goffmat', Goff*torch.ones(self.input_batch*self.num_pixel, xbars_row, 1, XBAR_ROW_SIZE, 1))
                G_real0 = (self.xbars[0]*(Gon - Goff)/Nstates_slice + Goff)
                G_real_scaled0 = (G_real0-Goff)/(Gon-Goff)                
                self.register_buffer('G_real_flatten0', G_real_scaled0.permute(0,1,3,2).reshape(xbars_row,xbars_col,XBAR_ROW_SIZE*XBAR_COL_SIZE))
                self.G_real_flatten0=self.G_real_flatten0.unsqueeze(3).expand(self.input_batch*self.num_pixel, xbars_row,xbars_col, XBAR_ROW_SIZE*XBAR_COL_SIZE, 1)
                
                G_real1 = (self.xbars[1]*(Gon - Goff)/Nstates_slice + Goff)
                G_real_scaled1 = (G_real1-Goff)/(Gon-Goff)                
                self.register_buffer('G_real_flatten1', G_real_scaled1.permute(0,1,3,2).reshape(xbars_row,xbars_col,XBAR_ROW_SIZE*XBAR_COL_SIZE))
                self.G_real_flatten1= self.G_real_flatten1.unsqueeze(3).expand(self.input_batch*self.num_pixel, xbars_row,xbars_col, XBAR_ROW_SIZE*XBAR_COL_SIZE, 1)

        else:
            self.shift_add_bit_stream= self.shift_add_bit_stream.expand((2, self.input_batch*self.num_pixel, xbars_row, xbars_col, 
                                                                                      XBAR_COL_SIZE//self.bit_slice_num, self.bit_stream_num)).transpose(4,5)
            self.shift_add_bit_slice=self.shift_add_bit_slice.expand((2, self.input_batch*self.num_pixel, xbars_row, xbars_col, 
                                                                                    XBAR_COL_SIZE//self.bit_slice_num, self.bit_slice_num))
            self.register_buffer('output_reg', torch.zeros(2, self.input_batch*self.num_pixel, xbars_row, xbars_col, self.bit_stream_num, XBAR_COL_SIZE//self.bit_slice_num))
            if self.ind == True:
                self.register_buffer('output_analog', torch.zeros(2, self.input_batch*self.num_pixel, xbars_row, xbars_col, XBAR_COL_SIZE))
                self.register_buffer('Goffmat', Goff*torch.ones(2, self.input_batch*self.num_pixel, xbars_row, 1, XBAR_ROW_SIZE, 1))
                G_real0 = (self.xbars[0]*(Gon - Goff)/Nstates_slice +Goff)
                G_real_scaled0 = (G_real0-Goff)/(Gon-Goff)                
                self.register_buffer('G_real_flatten0', G_real_scaled0.permute(0,1,3,2).reshape(xbars_row,xbars_col,XBAR_ROW_SIZE*XBAR_COL_SIZE))
                self.G_real_flatten0= self.G_real_flatten0.unsqueeze(3).expand(self.input_batch*self.num_pixel, xbars_row,xbars_col, XBAR_ROW_SIZE*XBAR_COL_SIZE, 1)                

                G_real1 = (self.xbars[1]*(Gon - Goff)/Nstates_slice +Goff)
                G_real_scaled1 = (G_real1-Goff)/(Gon-Goff)                
                self.register_buffer('G_real_flatten1', G_real_scaled1.permute(0,1,3,2).reshape(xbars_row,xbars_col,XBAR_ROW_SIZE*XBAR_COL_SIZE))
                self.G_real_flatten1= self.G_real_flatten1.unsqueeze(3).expand(self.input_batch*self.num_pixel, xbars_row,xbars_col, XBAR_ROW_SIZE*XBAR_COL_SIZE, 1)                                
                
        return Conv2d_mvm_function.apply(self.flatten_binary_input, self.flatten_input_sign_temp, self.flatten_input_sign_xbar, self.output, self.bias_addr, 
                                         self.weight_row, self.weight_col, self.output_row, self.output_col, self.input_pad, self.input_batch, self.pos,
                                         self.neg, self.xbars, self.zero_mvmtensor, self.shift_add_bit_stream, self.shift_add_bit_slice, self.output_reg, 
                                         self.output_analog, self.Goffmat, self.G_real_flatten0, self.G_real_flatten1, self.G_real0, self.G_real1, self.model,  
                                         self.tile_row, self.tile_row, self.bias, self.stride, self.padding, self.dilation, self.groups, self.bit_slice, self.bit_stream, 
                                         self.weight_bits, self.weight_bit_frac, self.input_bits, self.input_bit_frac, self.adc_bit, self.acm_bits, 
                                         self.acm_bit_frac, self.ind, self.loop, self.tile_row, self.tile_col)


class Linear_mvm_function(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, bit_slice=2, bit_stream=1, weight_bits=16, weight_bit_frac=-1, input_bits=16, input_bit_frac=-1, adc_bit=-1, acm_bits=16, acm_bit_frac=-1, ind=False, loop = True):

        if weight_bit_frac == -1:
            weight_bit_frac = weight_bits//4*3
        if input_bit_frac == -1:
            input_bit_frac = input_bits//4*3
        if acm_bit_frac == -1:
            acm_bit_frac = acm_bits//4*3      
        if adc_bit == -1:
            adc_bit = int(math.log2(XBAR_ROW_SIZE))
            if bit_stream != 1:
                adc_bit += bit_stream
            if bit_slice != 1:
                adc_bit += bit_slice
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        weight_channels_out = weight.shape[0]
        weight_channels_in = weight.shape[1]
#        weight_bias = torch.zeros(weight_channels_out+1, weight_channels_in).to(device)
#        weight_bias[:-1,:] = weight
        pos_weight = torch.clamp(weight, min=0)
        neg_weight = torch.clamp(weight, max=0).abs()


        pos_bit_slice_weight = bit_slicing(pos_weight, weight_bit_frac, bit_slice, weight_bits, device) ## v2: flatten weights --> fixed point --> bit slice -- v1
        neg_bit_slice_weight = bit_slicing(neg_weight, weight_bit_frac, bit_slice, weight_bits, device) ## 

        # bitsliced weight into 128x128 xbars 
        # xbar_row separates inputs --> results in a same column with different rows will be added later
        xbar_row = math.ceil(pos_bit_slice_weight.shape[0]/XBAR_ROW_SIZE)
        xbar_col = math.ceil(pos_bit_slice_weight.shape[1]/XBAR_COL_SIZE)

        weight_xbar = torch.zeros((2,xbar_row*XBAR_ROW_SIZE, xbar_col*XBAR_COL_SIZE)).to(device)
        weight_xbar[0,:pos_bit_slice_weight.shape[0], :pos_bit_slice_weight.shape[1]] = pos_bit_slice_weight
        weight_xbar[1,:neg_bit_slice_weight.shape[0], :neg_bit_slice_weight.shape[1]] = neg_bit_slice_weight

        xbars = torch.zeros((2,xbar_row, xbar_col, XBAR_ROW_SIZE, XBAR_COL_SIZE)).to(device)

        bit_slice_num = weight_bits//bit_slice
        bit_stream_num = input_bits//bit_stream

        bias_addr = [weight_channels_out//int(XBAR_COL_SIZE/bit_slice_num), weight_channels_out%int(XBAR_COL_SIZE/bit_slice_num)]      #####
        for i in range(xbar_row):
            for j in range(xbar_col):
                for k in range(2):
                    xbars[k,i,j] = weight_xbar[k,i*XBAR_ROW_SIZE:(i+1)*XBAR_ROW_SIZE, j*XBAR_COL_SIZE:(j+1)*XBAR_COL_SIZE]

        input_batch = input.shape[0]
        input_channels = input.shape[1]     # weight_channels_in == input_channels
        pos = torch.ones(input.shape).to(device)
        neg = pos.clone().fill_(0)      

        binary_input = torch.zeros(input_batch, xbars.shape[1]*XBAR_ROW_SIZE, bit_stream_num).to(device)
        input_sign_temp = torch.zeros(input_batch, xbars.shape[1]*XBAR_ROW_SIZE, bit_stream_num).to(device)
        input_sign_xbar = torch.zeros(input_batch, xbars.shape[1],XBAR_ROW_SIZE, bit_stream_num).to(device)

        
        if bit_stream > 1:
            input_sign = torch.where(input > 0, pos, neg).expand(bit_stream_num, -1, -1).permute(1,2,0)
            input_sign_temp[:,:input_sign.shape[1]] = input_sign
            input_sign_xbar = input_sign_temp.reshape(input_batch, xbars.shape[1],XBAR_ROW_SIZE, bit_stream_num)
            input.abs_()

#        binary_input[:,:input.shape[1]] = float_to_16bits_tensor(input, input_bit_frac, bit_stream, input_bits, device)   # batch x n x 16
        binary_input[:,:input.shape[1]] = float_to_16bits_tensor_fast(input, input_bit_frac, bit_stream, bit_stream_num, input_bits, device)   # batch x n x 16

        binary_input = binary_input.reshape((input_batch, xbars.shape[1], XBAR_ROW_SIZE, bit_stream_num))
        
        #initializations brought out of mvm_tensors, since they are only needed once for the output
        xbars_row = xbars.shape[1]
        xbars_col = xbars.shape[2]    
         
        zero_mvmtensor = torch.zeros(input_batch, xbars.shape[1],XBAR_ROW_SIZE, bit_stream_num).to(device)
        shift_add_bit_stream = torch.zeros(bit_stream_num) # input bits = 16
        for i in range(bit_stream_num):
            shift_add_bit_stream[i] = 2**(bit_stream*i)
        shift_add_bit_slice = torch.zeros(bit_slice_num) # 16bit / 2bit-slice
        for i in range(bit_slice_num):
            shift_add_bit_slice[-i-1] = 2**(bit_slice*i)        

        Gon = 1/100
        Goff = 1/600
        Nstates_slice = 2**bit_slice-1           
        if bit_stream ==1:
            shift_add_bit_stream[-1] *= -1        # last bit --> subtract
            shift_add_bit_stream = shift_add_bit_stream.expand((input_batch, xbars_row, xbars_col, XBAR_COL_SIZE//bit_slice_num, bit_stream_num)).transpose(3,4).to(device)
            shift_add_bit_slice = shift_add_bit_slice.expand((input_batch, xbars_row, xbars_col, XBAR_COL_SIZE//bit_slice_num, bit_slice_num)).to(device)
            output_reg = torch.zeros(input_batch, xbars_row, xbars_col, bit_stream_num, XBAR_COL_SIZE//bit_slice_num).to(device) # for 32-fixed  
            if ind == True:
                output_analog = torch.zeros(input_batch, xbars_row, xbars_col, XBAR_COL_SIZE).to(device)
                Goffmat = Goff*torch.ones(input_batch, xbars_row, 1, XBAR_ROW_SIZE, 1).to(device)
                G_real0 = (xbars[0]*(Gon - Goff)/Nstates_slice + Goff)
                G_real_scaled0 = (G_real0-Goff)/(Gon-Goff)                
                G_real_flatten0 = G_real_scaled0.permute(0,1,3,2).reshape(xbars_row,xbars_col,XBAR_ROW_SIZE*XBAR_COL_SIZE).to(device)
                G_real_flatten0 = G_real_flatten0.unsqueeze(3).expand(input_batch, xbars_row,xbars_col, XBAR_ROW_SIZE*XBAR_COL_SIZE, 1).to(device)
                
                G_real1 = (xbars[1]*(Gon - Goff)/Nstates_slice + Goff)
                G_real_scaled1 = (G_real1-Goff)/(Gon-Goff)                
                G_real_flatten1 = G_real_scaled1.permute(0,1,3,2).reshape(xbars_row,xbars_col,XBAR_ROW_SIZE*XBAR_COL_SIZE).to(device)
                G_real_flatten1 = G_real_flatten1.unsqueeze(3).expand(input_batch, xbars_row,xbars_col, XBAR_ROW_SIZE*XBAR_COL_SIZE, 1).to(device)                          
        else:
            shift_add_bit_stream = shift_add_bit_stream.expand((2, input_batch, xbars_row, xbars_col, XBAR_COL_SIZE//bit_slice_num, bit_stream_num)).transpose(4,5).to(device)
            shift_add_bit_slice = shift_add_bit_slice.expand((2, input_batch, xbars_row, xbars_col, XBAR_COL_SIZE//bit_slice_num, bit_slice_num)).to(device)
            output_reg = torch.zeros(2, input_batch, xbars_row, xbars_col, bit_stream_num, XBAR_COL_SIZE//bit_slice_num).to(device) 
            if ind == True:
                output_analog = torch.zeros(2, input_batch, xbars_row, xbars_col, XBAR_COL_SIZE).to(device)
                Goffmat = Goff*torch.ones(2, input_batch, xbars_row, 1, XBAR_ROW_SIZE, 1).to(device)
                G_real0 = (xbars[0]*(Gon - Goff)/Nstates_slice +Goff)
                G_real_scaled0 = (G_real0-Goff)/(Gon-Goff)                
                G_real_flatten0 = G_real_scaled0.permute(0,1,3,2).reshape(xbars_row,xbars_col,XBAR_ROW_SIZE*XBAR_COL_SIZE).to(device)
                G_real_flatten0 = G_real_flatten0.unsqueeze(3).expand(input_batch, xbars_row,xbars_col, XBAR_ROW_SIZE*XBAR_COL_SIZE, 1).to(device)                

                G_real1 = (xbars[1]*(Gon - Goff)/Nstates_slice +Goff)
                G_real_scaled1 = (G_real1-Goff)/(Gon-Goff)                
                G_real_flatten1 = G_real_scaled1.permute(0,1,3,2).reshape(xbars_row,xbars_col,XBAR_ROW_SIZE*XBAR_COL_SIZE).to(device)
                G_real_flatten1 = G_real_flatten1.unsqueeze(3).expand(input_batch, xbars_row,xbars_col, XBAR_ROW_SIZE*XBAR_COL_SIZE, 1).to(device)   
                
        if ind == True:
            xbars_out = mvm_tensor_ind(zero_mvmtensor, shift_add_bit_stream, shift_add_bit_slice, output_reg, output_analog, Goffmat, G_real_flatten0, G_real0, 
                                       model, loop, binary_input, input_sign_xbar, bias_addr, xbars[0], bit_slice, bit_stream, weight_bits, weight_bit_frac, 
                                       input_bits, input_bit_frac, adc_bit, acm_bits, acm_bit_frac, device) - \
                        mvm_tensor_ind(zero_mvmtensor, shift_add_bit_stream, shift_add_bit_slice, output_reg, output_analog, Goffmat, G_real_flatten1, G_real1, 
                                       model, loop, binary_input, input_sign_xbar, bias_addr, xbars[1], bit_slice, bit_stream, weight_bits, weight_bit_frac, 
                                       input_bits, input_bit_frac, adc_bit, acm_bits, acm_bit_frac, device)

        else:
            xbars_out = mvm_tensor(zero_mvmtensor, shift_add_bit_stream, shift_add_bit_slice, output_reg, binary_input, input_sign_xbar, bias_addr, xbars[0],
                                   bit_slice, bit_stream, weight_bits, weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bits, acm_bit_frac, device) - \
                        mvm_tensor(zero_mvmtensor, shift_add_bit_stream, shift_add_bit_slice, output_reg, binary_input, input_sign_xbar, bias_addr, xbars[1], 
                                   bit_slice, bit_stream, weight_bits, weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bits, acm_bit_frac, device)

        output = xbars_out[:, :weight_channels_out]
 
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias)

        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None 

class Linear_mvm(nn.Module):
    def __init__(self, input_features, output_features, bias=True, bit_slice = 2, bit_stream = 1, weight_bits=16, weight_bit_frac=-1, input_bits=16, input_bit_frac=-1, adc_bit=-1, acm_bits=16, acm_bit_frac=-1, ind = False, loop = True):
        super(Linear_mvm, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
            self.bias.data.uniform_(-0.1, 0.1) 
        else:
            self.register_parameter('bias', None)

        self.bit_slice = bit_slice
        self.bit_stream = bit_stream
        self.weight_bits =weight_bits
        self.weight_bit_frac = weight_bit_frac
        self.input_bits = input_bits
        self.input_bit_frac = input_bit_frac
        self.adc_bit = adc_bit
        self.acm_bits = acm_bits
        self.acm_bit_frac = acm_bit_frac
        self.ind = ind
        self.loop = loop

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return Linear_mvm_function.apply(input, self.weight, self.bias, self.bit_slice, self.bit_stream, self.weight_bits, self.weight_bit_frac, self.input_bits, self.input_bit_frac, self.adc_bit, self.acm_bits, self.acm_bit_frac, self.ind, self.loop)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )
