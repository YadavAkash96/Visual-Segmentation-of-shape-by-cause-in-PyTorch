import torch
import torch.nn as nn

    
def double_conv(input_channels,output_channels):
    conv = nn.Sequential(
        nn.Conv2d(input_channels,output_channels,kernel_size=3, padding=(1,1),bias=0),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels,output_channels,kernel_size=3, padding=(1,1),bias=0),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True)
        )
    
    return conv

class UNet(nn.Module):
    def __init__(self,input_channels=1,output_channels=1,filters=[64,128,256,512]):
        super(UNet,self).__init__()
         
        self.max_pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_list = nn.ModuleList()
        self.upconv_transpose = nn.ModuleList()
        self.up_conv_list = nn.ModuleList()
        
        for filter in filters:
            self.down_conv_list.append(double_conv(input_channels, filter))
            input_channels=filter
        
        #dropout layer
        #self.dropout = nn.Dropout2d(p=0.1)
        
        #bottelneck layer / last layer in down sample
        self.bottelneck_conv = double_conv(input_channels,1024)
        
        #print(self.down_conv_list)
        #print(self.bottelneck_conv)
        
        for filter in reversed(filters):
            self.upconv_transpose.append(nn.ConvTranspose2d(in_channels=filter*2,
                                        out_channels=filter, 
                                        kernel_size=2,
                                        stride=2
                                        ))
            
        #print(self.upconv_transpose)
        
        for filter in reversed(filters):
            self.up_conv_list.append(double_conv(filter*2, filter))
        
        #print(self.up_conv_list)
        
        #output layer single channel segmentation
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        
    
    #overriding inbuilt forward function
    def forward(self, image):
        #downsampling / encoder
        #p=image
        skip_connection = []
        for down_conv in self.down_conv_list:
            l = down_conv(image)
            #print("down conv layer:{}".format(l.size()))
            skip_connection.append(l)
            image = self.max_pool_layer(l)
            #p = self.dropout(p) 
            

        #dropoutlayer = self.dropout(p)            
        #bottelneck
        upconv = self.bottelneck_conv(image)
        skip_connection=skip_connection[::-1] #reversed, for tracking and merge skip layers 4--->1 
        
        #Decoder / Upsampling
        layeridx=0
        for up_conv, upconv_trans in zip(self.up_conv_list,self.upconv_transpose):
            #print("up conv layer:{} : {}".format(layeridx+1,skip_connection[layeridx].size()))
            transpose = upconv_trans(upconv)
            upconv = up_conv(torch.cat([skip_connection[layeridx],transpose],1))
            #if(layeridx!=3):
            #   dropout = self.dropout(upconv)
                #upconv=dropout
            #else:
                #dropout = upconv                
            layeridx+=1
        
        #print("final upconv:{}".format(upconv.size()))
            
        #print(bottelneck.size())
        #print(skip_connection)
        #dropout = nn.Dropout2d(p=0.2)(upconv)
        output_layer = self.final_conv(upconv)
        #print(output_layer.size())
        return output_layer
 