# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:36:23 2024

@author: Akash
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class OccludedObjectDataset(Dataset):
    def __init__(self,image_dir,mask_dir,transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir),key=lambda x: int(x.split('.')[0]))
        self.kernel_size = 3
        #self.images = self.images[:1000]
        
    def __len__(self):
        return len(self.images)
    

    
    def __getitem__(self, index):
        #check the order or images if not already sorted then use the sort function
        self.img_path = os.path.join(self.image_dir, self.images[index])
        self.mask_path = os.path.join(self.mask_dir, self.images[index])
        image = cv2.blur(cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE),(self.kernel_size, self.kernel_size))
        
        
        mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        
        image = np.round((image/255.0),4)
        mask = np.round((mask/255.0),4)
        
        image = np.expand_dims(image, axis=0) #(1,512,512)
        mask = np.expand_dims(mask, axis=0)
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        return image, mask
        