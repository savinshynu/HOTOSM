"""
This module has different working parts of the UNET architecture:
1. Two 3*3 convolutions followed by Relu - DoubleConv
2. Downsampling poolin step - DownSample
3. Upsampling by 

"""

import torch
import torch.nn as nn

"""
Interesting thing about Pytorch
All the layers are defined in the __init__ function and
all the feed forward part is defined in the forward function

mod_ob = Modelname()
mod_ob(x) >> will call model._call__(x) >> this will self.forward(x) function for the forward propogation
mod_ob(x) has different advantages as it tracks parameters, gradients etc compared to just calling self.forward (not preferred)
"""

class DoubleConv(nn.Module): # new class inheriting from the nn.Module base class
    """
    Double convolution step
    """
    def __init__(self, in_channels, out_channels):
        super().__init__() #initiating everything from the base class
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # padding is valid in the actual architecture
            nn.ReLU(inplace=True), # inplace=True will automatically updates the activation units
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    """
    Combination of double convolution and pooling
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # applying maxpooling to reduce the image size
    
    def forward(self, x):
        down = self.conv(x) # this will call teh DoubleConv class
        p = self.pool(down)

        return down, p


class UpSample(nn.Module):
    """
    Combination of transpose 2 D convolution, concatenation of (downsampled and 
    upsampled images) and 2 normal convolutions.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Basically mulitply each element of image with the 
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1) # transpose convolution
        #print(f"after transpose conv: {x1.shape}")
        #print(f"skip array shape: {x2.shape}")

        # The combination of low level features from downsampling and high level features from the upsampling
        # allows better localization
        x = torch.cat([x1, x2], 1) # concatenated along the frequency axis

        return self.conv(x) # apply a double convolution after concatenating

