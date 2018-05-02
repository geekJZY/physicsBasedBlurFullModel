import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision
import math
from PIL import Image
import numpy as np
from matplotlib import pyplot
from torch.autograd import Variable

def KernelGene( x, y, u ,v):
    exp_time = 0.5
    if u == 0 and v == 0:
        if x == 0 and y == 0:
            return 0.5
        else:
            return 0
    norm = math.sqrt(v*v+u*u)
    if u == 0:
        if y*v<0 or y/v > exp_time:
            return 0
        elif x == 0:
            return 1/(2*exp_time*norm)
        else:
            return 0;
    if v == 0:
        if x*u<0 or x/u > exp_time:
            return 0
        elif y == 0:
            return 1/(2*exp_time*norm)
        else:
            return 0;
    if u*x < 0 or y*v < 0 or x/u > exp_time or y/v > exp_time:
        return 0
        
    if abs((-x*v + y*u)/norm) < 0.5:
        return 1/(2*exp_time*norm)
    else:
        return 0


class flowToKernel(nn.Module):
    def __init__(self):
        super(flowToKernel, self).__init__()
        self.KernelInputTensor = torch.FloatTensor(1,1089,33,33)
        KernelOne = np.ones((33,33), dtype = float)
        for u in range(33):
            for v in range(33):
                for i in range(33):
                    for j in range(33):
                        KernelOne[i,j] = float(KernelGene(j - 16, 16 - i, u-16, v-16))
                self.KernelInputTensor[0, :, v, u] = torch.from_numpy(np.reshape(KernelOne/(2*np.sum(KernelOne)), -1))
        self.KernelInputTensor = Variable(self.KernelInputTensor, requires_grad=False).cuda()
#         self.enlarge = nn.Upsample(scale_factor=4, mode='nearest')
    def forward(self, flow1, flow2):
        """flow1 is optical flow from t-1 to t, flow2 is optical flow from t+1 to t"""
        Kernels1 = F.grid_sample(self.KernelInputTensor, flow1, padding_mode='border')
        Kernels2 = F.grid_sample(self.KernelInputTensor, flow2, padding_mode='border')
        return Kernels1 + Kernels2
