# Visual-Segmentation-of-shape-by-cause-in-PyTorch

**Objective:** In a virtual painting task, participants indicated which surface ridges appeared to be caused by the hidden object and which were due to the drapery.
**Source:** "The Veiled Virgin illustrates visual segmentation of shape by cause" (https://doi.org/10.1073/pnas.1917565117)


I have used UNet’s original paper to understand and build UNet architecture, although few changes have been made to the original architecture of UNet for better performance on our challenge. Batch Normalization and Dropout layers have been added after every set of conv layer, to normalize data and to make model less bias towards training features.

![image](https://github.com/AkashWelkin/Visual-Segmentation-of-shape-by-cause-in-PyTorch/assets/32175280/2c71a5ba-1fd1-4a08-8942-1a88574310d2)
          Source: https://doi.org/10.48550/arXiv.1505.04597

**Dataset:**
Normalized data between 0 – 1.
Applied blur filter at first layer for input images.
Data Augmentation: divided images in 4 quarters and fed to UNet (0.05% of dataset)
**Train module:**
Batch size:4 (max supported by cip pool GPU)
Epochs: 30 (validation loss was getting saturated after 15-18 epochs)
Learning Rate: 1e-4 to 1e-6
Loss Function: BCE+Logitloss. Also tried DiceBCE,MSE and Huberloss(didn’t work well)
Optimizer: Adam/RMSEProp
Scheduler: to adjust learning rate on plateau

![image](https://github.com/AkashWelkin/Visual-Segmentation-of-shape-by-cause-in-PyTorch/assets/32175280/10c08032-9823-4218-86c8-501aa89104d6)

**Results:**
**without binary labelling:**
![image](https://github.com/AkashWelkin/Visual-Segmentation-of-shape-by-cause-in-PyTorch/assets/32175280/c0a59d69-6479-4d8c-80ba-029f6be32c1c)
**with binary labelling:**
![image](https://github.com/AkashWelkin/Visual-Segmentation-of-shape-by-cause-in-PyTorch/assets/32175280/c399b96a-96ef-4dea-a796-34eccc7375df)




 
