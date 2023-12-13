import torch 
import torch.nn as nn
from Models.CNR_blocks import conv_norm_relu


class Build_UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Down Convolution
        self.max_pool3d = nn.MaxPool3d(kernel_size = 2,stride = 2)
        self.num_classes = num_classes
        self.down_convolution_1 = conv_norm_relu(3,64)
        self.down_convolution_2 = conv_norm_relu(64,128)
        self.down_convolution_3 = conv_norm_relu(128,256)
        self.down_convolution_4 = conv_norm_relu(256,512)
        self.down_convolution_5 = conv_norm_relu(512,1024)        
          
        # Up Convolution
        self.up_transpose_1 = nn.ConvTranspose3d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_convolution_1 = conv_norm_relu(1024,512)
        self.up_transpose_2 = nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_convolution_2 = conv_norm_relu(512,256)
        self.up_transpose_3 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_convolution_3 = conv_norm_relu(256,128)
        self.up_transpose_4 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_convolution_4 = conv_norm_relu(128,64)
        
        
        self.out = nn.Conv3d(
            in_channels=64, out_channels=self.num_classes, 
            kernel_size=1
        ) 
#         shape of out TENSOR : torch.Size([1, 4, 128, 128, 128])
# [dim0,dim1,dim2,dim3]
#         So we must put softmax operation in dim 1
        self.out_softmax = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        """ Encoder"""
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool3d(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool3d(down_3)
        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool3d(down_5)
        down_7 = self.down_convolution_4(down_6)
        down_8 = self.max_pool3d(down_7)
        down_9 = self.down_convolution_5(down_8)        
        # *** DO NOT APPLY MAX POOL TO down_9 ***
        
        """ Decoder"""
        up_1 = self.up_transpose_1(down_9)
        x = self.up_convolution_1(torch.cat([down_7, up_1], 1))
        
        up_2 = self.up_transpose_2(x)
        x = self.up_convolution_2(torch.cat([down_5, up_2], 1))
        
        up_3 = self.up_transpose_3(x)
        x = self.up_convolution_3(torch.cat([down_3, up_3], 1))
        
        up_4 = self.up_transpose_4(x)
        x = self.up_convolution_4(torch.cat([down_1, up_4], 1))
        
        out = self.out(x)
#         print(type(out))
#         print(out.shape)
        
#         out = (out/>0).float()
        
        out_softmax = self.out_softmax(out)
        return out_softmax