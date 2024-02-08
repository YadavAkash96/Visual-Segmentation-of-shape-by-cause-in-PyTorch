# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 22:42:43 2024

@author: Akash
"""
import numpy as np
import cv2
from utils import create_dir
import torch
from model import UNet
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "metadata2/checkpoint.pth"
TEST_IMG_DIR = "Submission/test/3" #"Submission/test/2"
MANUAL_MASK_DIR = "manual/3"
SAVE_IMG_DIR = "Submission/predictions/3"
SAVE_IMG_NUM = 1
TEST_MASK_DIR = "dataset/val_mask/"
TRAIN_IMG_DIR = "dataset/train_images/"
TRAIN_MASK_DIR = "dataset/train_mask/"
VAL_IMG_DIR = "dataset/cropped_validation_images"
VAL_MASK_DIR = "dataset/cropped_validation_mask"
crop_h = 256
crop_w = 256
img_h = 512
img_w = 512

def resize_image(image, target_size):
    return cv2.resize(image, (target_size[0], target_size[1]))


def split_image(image):
    height, width = image.shape[-2:]
    half_height, half_width = height // 2, width // 2
    
    top_left = image[:half_height, :half_width]
    top_right = image[:half_height, half_width:]
    bottom_left = image[half_height:, :half_width]
    bottom_right = image[half_height:, half_width:]
    
    return top_left, top_right, bottom_left, bottom_right

def combine_quarters(top_left, top_right, bottom_left, bottom_right):
    
    top = np.hstack((top_left, top_right))
    bottom = np.hstack((bottom_left, bottom_right))
    full_image = np.vstack((top, bottom))
    return full_image


def getTestImages(test_image_dir,test_mask_dir=None):
    RGB_channel=0
    if test_image_dir==MANUAL_MASK_DIR:
        RGB_channel=1
        
    images = sorted(os.listdir(test_image_dir),key=lambda x: int(x.split('.')[0]))
    test_img_list = []
    for img in images[:10]:
        #check the order or images if not already sorted then use the sort function
        img_path = os.path.join(test_image_dir, img)
        if(RGB_channel):
            image = cv2.imread(img_path)
        else:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        image = cv2.blur(image,(3,3))
        image = np.round((image/255.0),4) # (512, 512)
        if test_mask_dir:
            mask_path = os.path.join(test_mask_dir, img)
            mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = np.round((mask/255.0),4)
            
                            
            test_img_list.append((image,mask))
        else:
            test_img_list.append((image,None))
    
    return test_img_list

def calculate_accuracy(test_img_mask_list,pred_masks):
    test_distances = []
    pred_distances = []
    for idx in range(len(test_img_mask_list)):
        test_img = np.array(test_img_mask_list[idx][0],np.float32)
        test_mask =  np.array(test_img_mask_list[idx][1],np.float32)
        
        test_mask = np.where(test_img==0,0,test_mask)
        pred_mask = np.where(test_img==0,0,pred_masks[idx])
        test_distance = np.abs(test_img-test_mask)
        pred_distance = np.abs(test_img-pred_mask)
        test_distances.append(np.mean(test_distance))
        pred_distances.append(np.mean(pred_distance))
    mse = mean_squared_error(test_distances, pred_distances)
    rmse = np.sqrt(mse)
    print("Accuracy: RMSE: ",rmse)
   

def validationCurve(file_name):
    train_valid_loss = pd.read_csv(file_name)
    plt.plot(range(len(train_valid_loss)), train_valid_loss['train_loss'],label='Training_Loss',marker='o')
    plt.plot(range(len(train_valid_loss)), train_valid_loss['val_loss'],label='Validation_loss',marker='o')
    #remove sort function, definition has been updated
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation loss over number of epochs')
    plt.legend()
    plt.show()
    
    

if __name__ == "__main__":
    
    if not os.path.exists("results"):
        create_dir("results")
    
    model = UNet()
    model.to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)["state_dict"])
    model.eval()
    
    test_img_list = getTestImages(TEST_IMG_DIR)
    manual_mask_list = getTestImages(MANUAL_MASK_DIR)
    pred_masks = []
    
    for idx, (test_img,img_mask) in enumerate(tqdm(test_img_list)):
        manual_mask = manual_mask_list[idx][0]      
        top_left, top_right, bottom_left, bottom_right = split_image(test_img)
        
        # Resize quarters to 512x512
        top_left_resized = resize_image(top_left, (img_h, img_w))
        top_right_resized = resize_image(top_right, (img_h, img_w))
        bottom_left_resized = resize_image(bottom_left, (img_h, img_w))
        bottom_right_resized = resize_image(bottom_right, (img_h, img_w))
        
        # Convert the resized quarters to PyTorch tensor and move to device
        top_left_tensor = torch.from_numpy(top_left_resized.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
        top_right_tensor = torch.from_numpy(top_right_resized.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
        bottom_left_tensor = torch.from_numpy(bottom_left_resized.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
        bottom_right_tensor = torch.from_numpy(bottom_right_resized.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
        '''
        # Get predictions for image without spliting in parts
        with torch.no_grad():
            
            #torch.from_numpy(test_img.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
            pred_mask = model(test_img.to(DEVICE))
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = pred_mask.squeeze(0).squeeze(0).cpu().numpy()
            #pred_mask = pred_mask<0.5
            #pred_mask = np.array(pred_mask,dtype=np.float32)

        '''         
        with torch.no_grad():
            pred_top_left = model(top_left_tensor)
            pred_top_left = torch.sigmoid(pred_top_left)
            pred_top_left = (pred_top_left.squeeze().cpu().numpy() * 255)  # Convert to numpy and scale
            
            pred_top_right = model(top_right_tensor)
            pred_top_right = torch.sigmoid(pred_top_right)
            pred_top_right = (pred_top_right.squeeze().cpu().numpy() * 255)  # Convert to numpy and scale
            
            pred_bottom_left = model(bottom_left_tensor)
            pred_bottom_left = torch.sigmoid(pred_bottom_left)
            pred_bottom_left = (pred_bottom_left.squeeze().cpu().numpy() * 255)  # Convert to numpy and scale
            
            pred_bottom_right = model(bottom_right_tensor)
            pred_bottom_right = torch.sigmoid(pred_bottom_right)
            
            pred_bottom_right = (pred_bottom_right.squeeze().cpu().numpy()*255)  # Convert to numpy and scale
        # Resize predicted quarters to 256x256
        pred_top_left_resized = resize_image(pred_top_left, (crop_h, crop_w))
        pred_top_right_resized = resize_image(pred_top_right, (crop_h, crop_w))
        pred_bottom_left_resized = resize_image(pred_bottom_left, (crop_h, crop_w))
        pred_bottom_right_resized = resize_image(pred_bottom_right, (crop_h, crop_w))
        
        # Combine quarters to reconstruct the original image size
        reconstructed_image = combine_quarters(pred_top_left_resized, pred_top_right_resized, pred_bottom_left_resized, pred_bottom_right_resized)
    
        reconstructed_image = reconstructed_image/255.0
        #set desired threshold to get binary segmentation
        reconstructed_image = np.where(reconstructed_image>=0.35,reconstructed_image,0)
        pred_mask = np.array(reconstructed_image,dtype=np.float32)
        pred_masks.append(pred_mask/255.0)
        test_img = test_img*255
        manual_mask = manual_mask*255
        
        # can skip below lines and use plt.imshow(pre_mask,cmap='gray)
        test_img_rgb = cv2.cvtColor(test_img.astype(np.float32), cv2.COLOR_GRAY2RGB)
        pred_mask_rgb = cv2.cvtColor(pred_mask.astype(np.float32), cv2.COLOR_GRAY2RGB)
        cat_images = np.concatenate([pred_mask_rgb], axis=1)
        cv2.imwrite(f"{SAVE_IMG_DIR}/{idx+SAVE_IMG_NUM}.png", cat_images)
        
    #to get accuracy using RMSE calculation to calculate comparative distance between cloth and distance        
    #calculate_accuracy(test_img_list, pred_masks)


#training model will generate train-val loss, which can be plotted here
#validationCurve('metadata/loss_values_final.csv')
    
    
    
    
    
    
    
    