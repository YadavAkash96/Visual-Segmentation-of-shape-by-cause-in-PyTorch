# Visual-Segmentation-of-shape-by-cause-in-PyTorch

**Source:** [The Veiled Virgin illustrates visual segmentation of shape by cause] (https://doi.org/10.1073/pnas.1917565117)

**Objective:** In a virtual painting task, participants indicated which surface ridges appeared to be caused by the hidden object and which were due to the drapery.
The goal of this project was to implement the paper and offer insights into whether a Deep Neural Network (DNN) can outperform humans in producing results.

![image](https://github.com/AkashWelkin/Visual-Segmentation-of-shape-by-cause-in-PyTorch/assets/32175280/49f3cf76-54fe-4f2f-bebf-41de5b86c06a)

# Followed in Three parts:
### 1. Data Gathering and Generating
### 2. Calculate distance and Generate Ground truth
### 3. uNet Training


## 1. Data Gathering and Generating
**About**: Select 100 (can be taken more) 3D objects, scale to a unit sphere, drape them with cloth using physics simulation, and store the final stabilized cloth rendering.
###   Methodology:

  **Object Selection:** Link: https://huggingface.co/datasets/ShapeNet/ShapeNetCore (fill in details and request approval to access data).
                        The curation process involved selecting 100 3D objects from a vast repository containing more than 25,000 diverse and varied 3D models.
                        Objects were chosen based on their complexity and geometric diversity to ensure a robust simulation.
                        Store the objects with file name from 1 to 100 with .obj as extension within Object folder.

  **Scaling to Unit Sphere:** The Blender tool was utilized to uniformly scale selected 3D objects to fit within a unit sphere.
                              The normalization for objects was done using a custom Python script (python script), takes less than a minute for 100 objects.
  
  **Cloth Draping (Physics Simulation):** Blender's built-in physics simulation engine was employed to realistically drape the scaled 3D objects with cloth.
                                     The simulation took into account parameters such as cloth material properties, air viscosity, collision detection with the
                                     objects, etc.
  
  **Rendering and exporting the file(.obj):** The final step involved rendering the cloth-draped 3D objects within Blender to produce visually appealing and 
                                              realistic results.
                                              Get rid of the object underneath the cloth drape and export to be saved with file name from 1 to 100 with .obj as 
                                              extension within Cloth folder



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




 
