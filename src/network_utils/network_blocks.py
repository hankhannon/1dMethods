import torch
import torch.nn as nn

from utils import *

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
                 activation = nn.LeakyReLU(),
                 batch_norm : bool = True,
                 transpose : bool = False,
                 causal : bool = False):
        
        super(Conv1dBlock, self).__init__()
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
        
        if activation is not None:
            self.ops.append(activation)
        
        if batch_norm:
            self.ops.append(nn.BatchNorm1d(out_channels))
        
    def forward(self, input):
        return self.ops(input)
    
if __name__ == '__main__':

    X = torch.ones(32, 2, 1_000) #Batch, Channels, Length

    conv_01 = Conv1dBlock(2, 4, 5, 2)

    print(conv_01(X).shape) #expecting 32, 4, 500
        


        
