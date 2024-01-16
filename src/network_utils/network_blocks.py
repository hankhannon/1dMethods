import torch
import torch.nn as nn

from utils import *

ACTIVATION = {'relu': nn.ReLU(), 
              'leaky_relu': nn.LeakyReLU(), 
              'sigmoid': nn.Sigmoid(), 
              'tanh': nn.Tanh(), 
              'selu': nn.SELU(), 
              'linear': None}

class Conv1dBlock(nn.Module):
    '''
    Conv1d Block, performs a convolution, activation, and batchnorm. 
    Choice of normal conv1d or transpose conv1d, as well as 'causal' padding.
    '''
    def __init__(self, 
                 in_channels : int, 
                 out_channels : int, 
                 kernel_size : int = 3, 
                 stride : int = 1,
                 activation : str = 'leaky_relu',
                 batch_norm : bool = True,
                 transpose : bool = False,
                 causal : bool = False):
        
        super(Conv1dBlock, self).__init__()
        assert activation in ACTIVATION.keys()

        self.ops = nn.Sequential()
        if transpose:
            if kernel_size%2 != 0:
                kernel_size += 1
            
            conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)

            if causal:
                padding_size = int(kernel_size - stride)
                padding = Crop1d(padding_size, 0)
            else:
                padding_size = int((kernel_size - stride)//2)
                if padding_size * 2 != kernel_size-stride:
                    balance = 1
                else:
                    balance = 0
                padding = Crop1d(padding_size + balance, padding_size)
            
            self.ops.append(conv)
            self.ops.append(padding)

        else:
            if (kernel_size%2 == 0):
                kernel_size -= 1
                if kernel_size == 1:
                    kernel_size = 3
            
            conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)

            if causal:
                padding_size = (kernel_size - 1)
                padding = nn.ConstantPad1d((padding_size, 0), 0)
            else:
                padding_size = int((kernel_size - 1)//2)
                padding = nn.ConstantPad1d((padding_size, padding_size), 0)
            
            self.ops.append(padding)
            self.ops.append(conv)
        
        if activation != 'linear':
            self.ops.append(ACTIVATION[activation])
        
        if batch_norm:
            self.ops.append(nn.BatchNorm1d(out_channels))
        
    def forward(self, input):
        return self.ops(input)
    

class LoCon1d(nn.Module):
    def __init__(self,
                 in_channels : int,
                 out_channels : int,
                 seq_length : int,
                 kernel_size : int = 3,
                 stride : int = 1,
                 bias : bool = True,
                 causal : bool = False,
                 gain : float = 1.0):
        
        super(LoCon1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, seq_length, kernel_size) * gain
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, seq_length) * gain
            )
        else:
            self.register_parameter('bias', None)

        if causal:
            pad_size = int(self.kernel_size - 1)
            self.pad = nn.ConstantPad1d((pad_size, 0), 0)
        else:
            pad_size = int((self.kernel_size - 1)//2)
            self.pad = nn.ConstantPad1d((pad_size, pad_size), 0)

    
    def forward(self, input):

        unfolded = self.pad(input).unfold(2, self.kernel_size, self.stride)
        out = (unfolded.unsqueeze(1) * self.weight).sum([2,-1])
        if self.bias is not None:
            out += self.bias
        return out
    
class LoCon1dBlock(nn.Module):
    def __init__(self, 
                 in_channels : int,
                 out_channels : int,
                 seq_length : int,
                 kernel_size : int = 3,
                 stride : int = 1,
                 bias : bool = True,
                 causal : bool = False,
                 batch_norm : bool = True,
                 activation : str = 'leaky_relu'):
        
        super(LoCon1dBlock, self).__init__()
        assert activation in ACTIVATION.keys()

        self.ops = nn.Sequential()

        self.ops.append(
            LoCon1d(in_channels, out_channels, seq_length, kernel_size, stride, bias, causal, gain=torch.nn.init.calculate_gain(activation))
        )
        if activation != 'linear':
            self.ops.append(ACTIVATION[activation])
        if batch_norm:
            self.ops.append(nn.BatchNorm1d(out_channels))
        
    def forward(self, input):
        return self.ops(input)
    
if __name__ == '__main__':

    X = torch.ones(32, 2, 1_000) #Batch, Channels, Length

    conv_01 = Conv1dBlock(2, 4, 5, 2)
    conv_02 = LoCon1dBlock(2, 4, 1_000, 3, 1)

    print(conv_01(X).shape) #expecting 32, 4, 500
    print(conv_02(X).shape) #expecting 32, 4, 1_000
        


        
