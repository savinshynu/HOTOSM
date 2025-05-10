import torch
import torch.nn as nn

from unet_parts import DoubleConv, DownSample, UpSample

class UNet(nn.Module):
    """
    Putting all the UNET parts together
    """
    def __init__(self, in_channels, num_classes) :
        super().__init__()

        # Down sampling part
        self.down_conv1 = DownSample(in_channels, 64)
        self.down_conv2 = DownSample(64, 128)
        self.down_conv3 = DownSample(128, 256)
        self.down_conv4 = DownSample(256, 512)

        # Bottle neck which is another double convolution
        self.bottle_neck = DoubleConv(512, 1024)

        # Upsampling part
        self.up_conv1 = UpSample(1024, 512)
        self.up_conv2 = UpSample(512, 256)
        self.up_conv3 = UpSample(256, 128)
        self.up_conv4 = UpSample(128, 64)

        # Final output layer with no of channels = number of claases
        # Basically do a 1 by 1 convolution to control the number of output channels
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        # Down sampling part
        down1, p1 = self.down_conv1(x)
        down2, p2 = self.down_conv2(p1)
        down3, p3 = self.down_conv3(p2)
        down4, p4 = self.down_conv4(p3)
        
        #print("Down sampling shape")
        #print(down1.shape, p1.shape)
        #print(down2.shape, p2.shape)
        #print(down3.shape, p3.shape)
        #print(down4.shape, p4.shape)

        #Bottleneck
        b = self.bottle_neck(p4)

        #print(f"Bottleneck shape: {b.shape} ")
        
        # Upsampling part
        up1 = self.up_conv1(b, down4)
        up2 = self.up_conv2(up1, down3)
        up3 = self.up_conv3(up2, down2)
        up4 = self.up_conv4(up3, down1)

        #print("Upsampling shape:")
        #print(up1.shape)
        #print(up2.shape)
        #print(up3.shape)
        #print(up4.shape)

        #output
        out = self.out(up4)

        return out

if __name__ == "__main__":
    #double_conv = DoubleConv(3, 128)
    #print(double_conv)

    input_img = torch.rand((1, 3, 512, 512))
    model = UNet(3, 10)
    output  = model(input_img)

    print(output.shape)
