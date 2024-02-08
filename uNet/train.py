# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:17:37 2024

@author: Akash
"""

import time
import pandas as pd
import torch
import os
from tqdm import tqdm
import torch.nn as nn
from model import UNet
from utils import seeddata,create_dir,get_loaders,epoch_run_time


# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 30
PIN_MEMORY = True
IMAGE_HEIGHT = 512  # 1280 originally
IMAGE_WIDTH = 512  # 1918 originally
LOAD_MODEL = False
TRAIN_IMG_DIR = "dataset/new_train_images/"
TRAIN_MASK_DIR = "dataset/new_train_mask/"
VAL_IMG_DIR = "dataset/cropped_validation_images"
VAL_MASK_DIR = "dataset/cropped_validation_mask"
CHECKPOINT_PATH = "uNet/checkpoint.pth"

def train(loader, model, optimizer, loss_fn, scaler):
    epoch_loss = 0.0
    loop = tqdm(loader)
    
    model.train()
    for (data,label) in loop:
        data = data.to(device=DEVICE,dtype=torch.float32) #dtype = torch.float32 if usint this then can remove scaler
        label = label.to(device=DEVICE,dtype=torch.float32)
        
        assert data.device.type == torch.device(DEVICE).type
        assert label.device.type == torch.device(DEVICE).type
        
        #backpropagation
        optimizer.zero_grad()
        predictions = model(data)
        loss = loss_fn(predictions,label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # add/update epoch loss
        loop.set_postfix(loss=loss.item())
        epoch_loss+=loss.item()
    epoch_loss = epoch_loss/len(loader)
    return epoch_loss
        
        
def evaluate(loader, model, loss_fn):
    epoch_loss = 0.0
    loop = tqdm(loader)
    
    model.eval()
    with torch.no_grad():
        for data, label in loop:
            data = data.to(device = DEVICE,dtype=torch.float32)
            label = label.to(device = DEVICE,dtype=torch.float32)
            predictions = model(data)
            loss = loss_fn(predictions, label)
            epoch_loss+=loss.item()
            loop.set_postfix(loss=loss.item())
        epoch_loss=epoch_loss/len(loader)
    
    return epoch_loss

def verify_model_on_cuda(model):
    # Check if model parameters are on CUDA
    for param_tensor in model.state_dict():
        if model.state_dict()[param_tensor].device.type != DEVICE:
            print(f"Model parameter {param_tensor} is not on the expected device ({DEVICE}).")
            return False
    return True

if __name__ == "__main__":
    
    if DEVICE == 'cuda':
        print(torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        
    seeddata(42)
    
    if not os.path.exists("metadata"):
        create_dir("metadata")
   
    loss_log = pd.DataFrame(columns=['train_loss','val_loss'])
    train_loader, val_loader = get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR
                                           , BATCH_SIZE)
    model = UNet().to(DEVICE)
    if verify_model_on_cuda(model):
        print("Model is on CUDA.")
    else:
        print("Model is not on CUDA.")
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
    loss_fn = nn.BCEWithLogitsLoss()
    
    scaler = torch.cuda.amp.GradScaler()
    
    #training
    best_valid_loss = float("inf")
    patience=0
    for epoch in tqdm(range(NUM_EPOCHS)):
        start_time = time.time()
        print("\nTraining batch:")
        train_loss = train(train_loader, model, optimizer, loss_fn, scaler)
        print("\nValidation batch:")
        valid_loss = evaluate(val_loader, model, loss_fn)
        # Adjust learning rate using the scheduler
        scheduler.step(valid_loss) 
        
        if(patience>5):
            print("No improvement since last 5 consecutive epochs")
            loss_log.to_csv('metadata/loss_values.csv',index=False)
            break
        # save model
        loss_log = loss_log.append({'train_loss':train_loss,'val_loss':valid_loss},ignore_index=True)
        #to check trend of train and valid loss over different epochs
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {CHECKPOINT_PATH}"    
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict()
                }
            best_valid_loss = valid_loss
            torch.save(checkpoint, CHECKPOINT_PATH)
            loss_log.to_csv('metadata2/loss_values.csv',index=False)
            patience=0
        else:
            patience+=1
            
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_run_time(start_time, end_time)
        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}'
    loss_log.to_csv('metadata/loss_values.csv',index=False)
    
    
    
    
        
        
        
        
        
        
        
    
    
