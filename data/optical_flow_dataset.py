import torch.utils.data as data
import os
import os.path
from scipy.ndimage import imread
import numpy as np
import glob


def default_loader(root, path_imgs):
    imgs = [os.path.join(root,path) for path in path_imgs]
    return [imread(img, mode="RGB").astype(np.float32) for img in imgs]

class ListDataset(data.Dataset):
    def __init__(self, root, path_list, transform=None,
                 co_transform=None, loader=default_loader):

        self.root = root
        self.path_list = path_list
        self.transform = transform
        self.co_transform = co_transform
        self.loader = loader

    def __getitem__(self, index):
        inputs = self.path_list[index]

        inputs = self.loader(self.root, inputs)
        if self.transform is not None:
            inputs[0] = self.transform(inputs[0])
            inputs[1] = self.transform(inputs[1])
        return inputs

    def __len__(self):
        return len(self.path_list)
    
def make_dataset(dir):
    '''Will search for triplets that go by the pattern '[name]_img1.png  [name]_img2.png '''
    images = []
    for flow_map in sorted(glob.glob(os.path.join(dir,'*_img1.png'))):
        flow_map = os.path.basename(flow_map)
        root_filename = flow_map[:-9]
        img1 = root_filename+'_img1.png'
        img2 = root_filename+'_img2.png'
        if not (os.path.isfile(os.path.join(dir,img1)) or os.path.isfile(os.path.join(dir,img2))):
            print("{} is not a file".format(os.path.join(dir,img1)))
            continue

        images.append([img1,img2])

    return images


def optical_flow_dataset(root, transform=None, target_transform=None,
                  co_transform=None):
    test_list = make_dataset(root)
    test_dataset = ListDataset(root, test_list, transform)

    return test_dataset