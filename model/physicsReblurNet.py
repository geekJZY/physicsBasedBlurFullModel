import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F
import flow_transforms
from .FlowNetS import flownets
from .OpticalFlowToKernel import flowToKernel
from .reblurWithKernel import reblurWithKernel
import numpy as np
import scipy.misc

class physicsReblurNet(nn.Module):
    
    def __init__(self):
        super(physicsReblurNet,self).__init__()
        
        # data range transform
        self.rectifyKernel = Variable(torch.ones(3,1,1,1), requires_grad=False).cuda()/2
        self.rectifyBias = Variable(torch.FloatTensor([0.089,0.068,0.05]), requires_grad=False).cuda()
        self.transbackKernel = Variable(torch.ones(3,1,1,1), requires_grad=False).cuda()*2
        self.transbackBias = Variable(torch.FloatTensor([-0.089,-0.068,-0.05]), requires_grad=False).cuda()*2
        
        # optical flow axis direction flip
        self.flipKernel = Variable(torch.ones(2,1,1,1), requires_grad=False).cuda()
        self.flipKernel[1] = -flipKernel[1]
        
        # create model
        network_data = torch.load('/scratch/user/jiangziyu/flownets_EPE1.951.pth.tar')
        model = flownets(network_data).cuda()
        # Load Kernel Calculation Module
        KernelModel = flowToKernel()
        # Load blurWithKernel Module
        BlurModel = reblurWithKernel()
        
    
    
    def forward(image0, image1, image2)
        # adjust input data range
        image0 = F.conv2d(image0,self.rectifyKernel,self.rectifyBias,groups=3)
        image1 = F.conv2d(image1,self.rectifyKernel,self.rectifyBias,groups=3)
        image2 = F.conv2d(image2,self.rectifyKernel,self.rectifyBias,groups=3)
        # optical flow
        input_var_1 = torch.autograd.Variable(torch.cat([image0, image1],1).cuda())
        input_var_2 = torch.autograd.Variable(torch.cat([image2, image1],1).cuda())
        # flip axis
        output_1 = F.conv2d(output_1, self.flipKernel, groups = 2)
        output_2 = F.conv2d(output_2, self.flipKernel, groups = 2)
        # kernel estiamte
        output_1 = model(input_var_1)
        output_2 = model(input_var_2)
        output_1 = torch.transpose(output_1, 1, 2)
        output_1 = torch.transpose(output_1, 2, 3)
        output_2 = torch.transpose(output_2, 1, 2)
        output_2 = torch.transpose(output_2, 2, 3)
        ImageKernels = KernelModel.forward((20/16) * output_1, (20/16) * output_2)
        # transform back and blur with kernels
        image1 = F.conv2d(image1,self.transbackKernel,self.transbackBias,groups=3)
        blurImg = BlurModel.forward(image1, ImageKernels) 

def flow2rgb(flow_map, max_value):
    _, h, w = flow_map.shape
#     print(flow_map)
    flow_map[:,(flow_map[0] == 0) & (flow_map[1] == 0)] = float('nan')
    rgb_map = np.ones((h,w,3)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map / max_value
    else:
        normalized_flow_map = flow_map / (np.abs(flow_map).max())
    rgb_map[:,:,0] += normalized_flow_map[0]
    rgb_map[:,:,1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[:,:,2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)