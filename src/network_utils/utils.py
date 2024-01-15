import torch.nn as nn

class Crop1d(nn.Module):
    def __init__(self, 
                 front_crop : int,
                 back_crop : int = 0):
        
        super(Crop1d, self).__init__()
        self.front_crop = front_crop
        self.back_crop = back_crop
    
    def forward(self, input):
        l = input.shape[-1]
        return input[:,:,self.front_crop:int(l - self.back_crop)]