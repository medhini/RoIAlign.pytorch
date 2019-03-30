#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader

from roi_align.roi_align import RoIAlign      # RoIAlign module
from roi_align.roi_align import CropAndResize # crop_and_resize module
# from image_loader import Image_Reader
from torch.utils.data import DataLoader

import json
import torch
import string
import pprint
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


# In[8]:


class ImageReader(Dataset):
    def __init__(self):
        self.__prepare_feat()

    def __prepare_feat(self):
        self.root_path = '/scratch/medhini2/paragraph_generation/dataset/'
        self.file_paths = os.listdir(self.root_path + '/resnet_features/')
            
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filename = self.file_paths[idx]
        feat = torch.load(os.path.join(self.root_path + '/resnet_features/', filename))
        box = torch.load(os.path.join(self.root_path + '/boxes/', filename))
        box = torch.FloatTensor(box)
        box_idx = len(box)
        if box_idx == 0:
            box = torch.zeros(2, 4)
        return filename, feat, box, box_idx
    
images = ImageReader()
image_data = DataLoader(dataset=images, num_workers=4, batch_size=1, shuffle=False, drop_last=False) #How to make it work for batch size >1


# In[ ]:


os.environ["CUDA_VISIBLE_DEVICES"] = str(2)
crop_height = 7
crop_width = 7
roi_dir = '/scratch/medhini2/paragraph_generation/dataset/RoI/'
for batch_idx, (filename, featuremap_data, boxes_data, boxes_idx) in enumerate(image_data):
    featuremap = Variable(featuremap_data, requires_grad=True).cuda()
    if boxes_idx[0]!=0:
        boxes = Variable(boxes_data, requires_grad=False).cuda()
        box_index = Variable(boxes_idx.type(torch.cuda.IntTensor), requires_grad=False).cuda()
        #RoIAlign layer
        roi_align = RoIAlign(crop_height, crop_width)
        crops = roi_align(featuremap, boxes[0], box_index)
        roi_file = os.path.join(roi_dir,filename[0])
        torch.save(crops.data,roi_file)
        


# In[ ]:




