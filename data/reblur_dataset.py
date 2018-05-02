import torch
from data.base_dataset import BaseDataset
from PIL import Image
import torchvision.transforms as transforms
import random
import os

class reblurDataSet(BaseDataset):
    """Reblur dataset."""

    def initialize(self, opt):
        self.opt = opt
        self.root_dir = opt.dataroot
        self.folders = sorted(os.listdir(self.root_dir))
        foldersLen = []
        foldersStart = []
        for folder in self.folders:
            tempList = sorted(os.listdir(os.path.join(self.root_dir, folder,"sharp")))
            foldersLen.append(len(tempList))
            foldersStart.append(int(tempList[0].split('.')[0]))
        self.foldersLen = foldersLen
        self.foldersStart = foldersStart
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)
        

    def __len__(self):
        return sum(self.foldersLen) - 2 * len(self.foldersLen)

    def __getitem__(self, offset):
        images = []
        sample = {}
        cnt = 0
        offset = offset + 1
        while offset > 0:
            offset = offset - self.foldersLen[cnt] + 2
            cnt = cnt + 1
        cnt = cnt - 1
        offset = offset + self.foldersLen[cnt] - 3
        for index in range(3):
            img_name = os.path.join(self.root_dir, self.folders[cnt], "blur"
                                ,str(self.foldersStart[cnt]+offset+index).zfill(6)+".png")
            sample['image'+str(index)] = Image.open(img_name).convert('RGB')
			sample['image'+str(index)] = sample['image'+str(index)].resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
        label_name = os.path.join(os.path.join(self.root_dir, self.folders[cnt], "sharp"
                                ,str(self.foldersStart[cnt]+offset+1).zfill(6)+".png"))
        sample['label'] = Image.open(label_name).convert('RGB')
		sample['label'] = sample['label'].resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
        sample = {key:self.transform(sample[key]) for key in sample}
        
        w = sample['label'].size(2)
        h = sample['label'].size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        sample = {key:sample[key][:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize] for key in sample}

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(sample['label'].size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            sample = {key:sample[key].index_select(2, idx) for key in sample}

        return sample
    
    def name(self):
        return 'reblurDataSet'
