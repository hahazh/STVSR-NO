import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image
import data.core_bicubic as core_bicubic
from torchvision import transforms
class Vid4(Dataset):  # load train dataset
    def __init__(self, data_dir, scale):
        super(Vid4, self).__init__()
        self.name_list = ('calendar','city','foliage','walk')
        self.seq_list = []
        for name in self.name_list:
            this_dir = os.path.join(data_dir,name)
            self.seq_list.append([os.path.join(this_dir,each) for each in sorted(os.listdir(this_dir))])
        self.scale = scale
       
   
    def load_img(self, image_list):

        HR = []
        for img in image_list:
            img_gt = Image.open(img).convert('RGB') 
            HR.append(img_gt)
        HR = [np.asarray(HR) for HR in HR]
        
        return HR

    def __getitem__(self, index):
        GT = self.load_img(self.seq_list[index]) 
        GT = np.asarray(GT).astype('float32')/255.0  # numpy, [T,H,W,C], stack with temporal dimension
        t = GT.shape[0]
        h = GT.shape[1]
        w = GT.shape[2]
        c = GT.shape[3]
   
        
        GT = torch.from_numpy(GT).permute(0,3,1,2).contiguous()
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        GT_pad = GT.new_zeros( t, c, ph, pw)
        GT_pad[:,:,:h,:w] = GT
     
        LRindex = [int(i) for i in range(GT.shape[0]) if i%2==0]
        LR = core_bicubic.imresize(GT_pad[ LRindex,:,:,:],sizes = (round(ph//self.scale),round(pw//self.scale)))
        crop_shape = (0,h,0,w)
        return GT,LR,self.name_list[index],crop_shape

    def __len__(self):
        return len(self.name_list)  # total video number. not image number
    


class Vid4_arb(Dataset):  # load train dataset
    def __init__(self, data_dir, scale):
        super(Vid4_arb, self).__init__()
        self.name_list = ('calendar','city','foliage','walk')
        self.seq_list = []
        for name in self.name_list:
            this_dir = os.path.join(data_dir,name)
            self.seq_list.append([os.path.join(this_dir,each) for each in sorted(os.listdir(this_dir))])
        self.scale = scale
       
   
    def load_img(self, image_list):

        HR = []
        for img in image_list:
            img_gt = Image.open(img).convert('RGB')
            HR.append(img_gt)
        HR = [np.asarray(HR) for HR in HR]
        
        return HR

    def __getitem__(self, index):
        GT = self.load_img(self.seq_list[index]) 
        GT = np.asarray(GT).astype('float32')/255.0  # numpy, [T,H,W,C], stack with temporal dimension
        t = GT.shape[0]
        h = GT.shape[1]
        w = GT.shape[2]
        c = GT.shape[3]
   
        
        GT = torch.from_numpy(GT).permute(0,3,1,2).contiguous()
       
     
        LRindex = [int(i) for i in range(GT.shape[0]) if i%2==0]
        LR = core_bicubic.imresize(GT[ LRindex,:,:,:],sizes = (round(h/self.scale),round(w/self.scale)))
        crop_shape = (h,w)
        return GT,LR,self.name_list[index],crop_shape

    def __len__(self):
        return len(self.name_list)  # total video number. not image number