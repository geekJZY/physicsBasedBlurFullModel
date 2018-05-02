import torch.nn as nn
import torch.nn.functional as F
import torch

class reblurWithKernel(nn.Module):
    def __init__(self):
        super(reblurWithKernel, self).__init__()
        self.padding = nn.ReplicationPad2d(16)
        self.indices = torch.autograd.Variable(torch.LongTensor(list(range(36)))).cuda()

    def forward(self, inputImg, Kernels):
        """Input image is in 1*C*H*W form"""
#         print(inputImg.size())
#         print(Kernels.size())
        batchs, channels, height, width = list(inputImg.size())
        output = torch.ones_like(inputImg).cuda()
        inputImg = self.padding(inputImg)
        for h in range(height//4):
            for w in range(width//4):
                temp = torch.index_select(inputImg, 2, self.indices + h*4)
                temp = torch.index_select(temp, 3, self.indices + w*4)
                
                kernel = (Kernels[0,:,h,w]).contiguous().view(1,1,33,33)
                kernel = torch.cat([kernel,kernel,kernel],0)
#                 print(temp.size())
#                 print(kernel.size())
#                 print(output[:,:,h*4:h*4+4,w*4:w*4+4].size())
#                 print(F.conv2d(temp, kernel, groups=3).size())
                output[:,:,h*4:h*4+4,w*4:w*4+4] = F.conv2d(temp, kernel, groups=3)
        return output
