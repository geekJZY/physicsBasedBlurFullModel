
# In[1]:


import time
import sys
import torch
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.metrics import PSNR, SSIM
from skimage.measure import compare_psnr


# ## Import dataloader and show it

# In[2]:

sys.argv += ['--dataroot', '/scratch/user/jiangziyu/train/',
             '--learn_residual', '--resize_or_crop', 'scale_width',
             '--fineSize', '256','--batchSize','4','--name','fullModelWithGANLoss','--model','pix2pix']

opt = TrainOptions().parse()


# In[3]:


#get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from data.full_model_dataset import fullModelDataSet 

dataset = fullModelDataSet()
dataset.initialize(opt)


# In[4]:


dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))


# ## define model and load pretrained weights

# In[5]:

import os
from torch.autograd import Variable
from collections import OrderedDict
from models import networks
from models import multi_in_networks

def load_network(network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
        network.load_state_dict(torch.load(save_path))
def save_network(network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if torch.cuda.is_available():
            network.cuda()

netG_deblur = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.gpu_ids, False,
                                      opt.learn_residual)
netG_blur = multi_in_networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.gpu_ids, False,
                                      opt.learn_residual)
use_sigmoid = opt.gan_type == 'gan'
netD = networks.define_D(opt.output_nc, opt.ndf,
                                  opt.which_model_netD,
                                  opt.n_layers_D, opt.norm, use_sigmoid, opt.gpu_ids, False)

load_network(netG_deblur, 'deblur_G', opt.which_epoch)
load_network(netG_blur, 'blur_G', opt.which_epoch)
load_network(netD, 'D', opt.which_epoch)
print('------- Networks deblur_G initialized ---------')
networks.print_network(netG_deblur)
print('-----------------------------------------------')

print('------- Networks deblur_D initialized ---------')
networks.print_network(netD)
print('-----------------------------------------------')
# ### Freeze layers

# In[6]:


def freeze_single_input(model,num_layers_frozen=19):

    ct=0
    for child in list(model.children())[0]:
        ct+=1
        if ct<num_layers_frozen:
            for param in child.parameters():
                param.requires_grad=False


    print("Total number of layers are:",ct,",number of layers frozen are:", num_layers_frozen)
    return model

def freeze_multi_input(model,num_layers_frozen=19):
    if num_layers_frozen < 2:
        pass
    for i,child in enumerate(list(model.children())[0].children()):
        if i == 3:
            break
        for param in child.parameters():
            param.requires_grad=False
    ct=0
    for child in list(list(model.children())[0].children())[3]:
        ct+=1
        if ct<num_layers_frozen-2:
            for param in child.parameters():
                param.requires_grad=False


    print("Total number of layers are:",ct+2,",number of layers frozen are:", num_layers_frozen)
    return model

netG_frozen_deblur= freeze_single_input(netG_deblur, num_layers_frozen=0)
netG_frozen_blur= freeze_multi_input(netG_blur, num_layers_frozen=50)
netD_frozen = freeze_single_input(netD,num_layers_frozen=50);

# ### Net training parameters

# In[8]:


num_epoch=100
num_workers=2
learning_rate=0.0001
lrd = 0.000002
transforms=None       #make data augmentation. For now using only the transforms defined above

# ### Cycle consistency loss

# In[9]:


import itertools
import util.util as util
import numpy as np
from models.losses import init_loss

"""Quote from the paper about the loss function: For all the experiments, we set λ = 10 in Equation 3.
We use the Adam solver [24] with a batch size of 1"""

cycle_consistency_criterion= torch.nn.L1Loss()
disLoss, _ = init_loss(opt, torch.cuda.FloatTensor)
#criterion= forward_cycle_consistency_criterion+backward_cycle_consistency_criterion()

#lambda_cycle is irrelevant for the moment as we use only cycle consistency loss as of now

optimizer = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, netG_frozen_deblur.parameters()),
filter(lambda p: p.requires_grad, netG_frozen_blur.parameters())), lr=learning_rate)


# ### Training

# In[10]:

def model_type_gpu(blur_net, deblur_net):
    num_gpus= torch.cuda.device_count()

    if num_gpus>1:
        print("more than 1 GPU detected...")
        netDeblur=torch.nn.DataParallel(deblur_net)
        netBlur=torch.nn.DataParallel(blur_net)

    elif num_gpus==1:
        print("A GPU detected...")
        netDeblur=deblur_net.cuda()
        netBlur=blur_net.cuda()

    else:
        pass

model_type_gpu(netG_frozen_deblur,netG_frozen_blur)      ##make the correct definition for the model


for epoch in range(num_epoch):
    for i, data in enumerate(dataloader):
        images0 = Variable(data['image0']).cuda()
        images1 = Variable(data['image1']).cuda()
        images2 = Variable(data['image2']).cuda()
        labels = Variable(data['label']).cuda()


        optimizer.zero_grad()

        #forward loss part 
        deblur_out0 = netG_frozen_deblur.forward(images0)
        deblur_out1 = netG_frozen_deblur.forward(images1)
        deblur_out2 = netG_frozen_deblur.forward(images2)
        blur_model_outputs_f = netG_frozen_blur.forward(deblur_out0, deblur_out1, deblur_out2)
        loss_unsupervise = cycle_consistency_criterion(blur_model_outputs_f, images1)
        loss_dis = disLoss.get_loss(netD_frozen, images1, deblur_out1, labels)
        loss = loss_unsupervise + loss_dis*0.0025
        #backward loss part
        
        
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 20 == 0:
            print("(epoch %d itr %d), unsupervised loss is %f, L1 loss is %f, loss is %f"% (epoch, i+1, loss_unsupervise.data[0], loss_dis.data[0], loss.data[0]))
            print("(epoch %d itr %d), unsupervised loss is %f, L1 loss is %f, loss is %f"% (epoch, i+1, loss_unsupervise.data[0], loss_dis.data[0], loss.data[0]), file=open("outputFullModelMultiGAN.txt", "a"))
         
        
    if epoch%2 ==0:    ##save deblur once every 2 epochs
        save_network(netG_deblur, 'deblur_G', opt.which_epoch)
        save_network(netG_deblur, 'deblur_G', epoch)
    if epoch>50:
        lr = learning_rate - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr        