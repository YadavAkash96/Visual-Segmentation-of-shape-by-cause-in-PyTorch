# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:24:03 2024

@author: Akash
"""

import os
import random
import numpy as np
from dataset import OccludedObjectDataset
from torch.utils.data import DataLoader
import torch



#seed data
def seeddata(seed_number):
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number) # setting random number generation seed for CPU
    torch.cuda.manual_seed(seed_number) # setting random number generation seed for GPU
    torch.backends.cudnn.deterministic = True

#create directory for saving model weights and checkpoints
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
#time calculation for each epoch
def epoch_run_time(start_time, end_time):
    time_diff = end_time - start_time
    time_mins = int(time_diff/60)
    time_secs  = int(time_diff - (time_mins*60))
    return time_mins, time_secs
    


def get_loaders(train_dir,mask_dir,val_dir,valmask_dir,batch_size,num_workers=2,pin_memory=True):
    
    train_ds = OccludedObjectDataset(image_dir=train_dir,mask_dir=mask_dir)
    val_ds = OccludedObjectDataset(image_dir=val_dir, mask_dir=valmask_dir)
    
    train_loader = DataLoader(
                    dataset=train_ds,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True
                    )
    valid_loader = DataLoader(
                    dataset=val_ds,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True
                    )
    return train_loader, valid_loader 
