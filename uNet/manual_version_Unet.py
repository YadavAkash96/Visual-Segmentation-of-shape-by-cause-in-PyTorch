# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 23:33:45 2024

@author: Akash
"""

import torch
import torch.nn as nn

def double_conv(input_channels,output_channels):
    conv = nn.Sequential(
        nn.Conv2d(input_channels,output_channels,kernel_size=3, padding='same',bias=0),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels,output_channels,kernel_size=3, padding='same',bias=0),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True)
        )
    
    return conv

class UNet(nn.Module):
    def __init__(self,input_channels=1,output_channels=1,filters=[64,128,256,512]):
        super(UNet,self).__init__()
         
        self.max_pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down_conv_1 = double_conv(1, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512,1024) # in orginal paper it's not exactly double layer for this line.
    
        self.dropout = nn.Dropout2d(p=0.2)
        
        self.upconv_transpose_1 = nn.ConvTranspose2d(in_channels=1024,
                                                   out_channels=512, 
                                                   kernel_size=2,
                                                   stride=2
                                                   )
        self.up_conv_1 = double_conv(1024, 512)
        
        self.upconv_transpose_2 = nn.ConvTranspose2d(in_channels=512,
                                                   out_channels=256, 
                                                   kernel_size=2,
                                                   stride=2
                                                   )
        self.up_conv_2 = double_conv(512, 256)
        
        self.upconv_transpose_3 = nn.ConvTranspose2d(in_channels=256,
                                                   out_channels=128, 
                                                   kernel_size=2,
                                                   stride=2
                                                   )
        self.up_conv_3 = double_conv(256, 128)
        
        self.upconv_transpose_4 = nn.ConvTranspose2d(in_channels=128,
                                                   out_channels=64, 
                                                   kernel_size=2,
                                                   stride=2
                                                   )
        self.up_conv_4 = double_conv(128, 64)
        
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        
    
    #overriding inbuilt forward function
    def forward(self, image):
        layer1 = self.down_conv_1(image)
        #print(layer1.size())
        pool1 = self.max_pool_layer(layer1)
        layer2 = self.down_conv_2(pool1)
        pool2 = self.max_pool_layer(layer2)
        layer3 = self.down_conv_3(pool2)
        pool3 = self.max_pool_layer(layer3)
        layer4 = self.down_conv_4(pool3)
        pool4 = self.max_pool_layer(layer4)
        
        ulayer = self.down_conv_5(pool4)
        
        
        print(layer4.size())
        print(ulayer.size())
        #upsampling / decoder
        upconv = self.upconv_transpose_1(ulayer)
        concat_skip_layer = torch.cat([layer4, upconv],1)
        concat_up_1 = self.up_conv_1(concat_skip_layer)
        
        dropout = self.dropout(concat_up_1)
        
        upconv = self.upconv_transpose_2(dropout)
        concat_skip_layer = torch.cat([layer3, upconv],1)
        concat_up_1 = self.up_conv_2(concat_skip_layer)
        
        dropout = self.dropout(concat_up_1)
        
        upconv = self.upconv_transpose_3(dropout)
        concat_skip_layer = torch.cat([layer2, upconv],1)
        concat_up_1 = self.up_conv_3(concat_skip_layer)
        
        dropout = self.dropout(concat_up_1)
        
        upconv = self.upconv_transpose_4(dropout)
        concat_skip_layer = torch.cat([layer1, upconv],1)
        concat_up_1 = self.up_conv_4(concat_skip_layer)
        
        dropout = nn.Dropout2d(p=0.5)(concat_up_1)
        print(concat_up_1.size())
        
        output_layer = self.final_conv(dropout)
        
        
        print(output_layer.size())
        return output_layer
     

if __name__ == "__main__":
    image = torch.rand((1,1,512,512))
    model = UNet()
    print(model(image).size())    
    
        
        
        
        
        