import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_channels, out_channels, kernel_size=3, padding=None, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2 if padding == None else padding, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. define multiple convolution and downsampling layers
        # 2. define full-connected layer to classify  
        
        def conv_stage(in_channels, out_channels):
            return nn.Sequential(
                conv_block(in_channels, out_channels),
                conv_block(out_channels, out_channels),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.dropout = nn.Dropout(0.3)
        self.stage1 = conv_stage(3, 64)
        self.stage2 = conv_stage(64, 128)
        self.stage3 = conv_stage(128, 256)

        self.fclinear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 10)
        )

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        # classification
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.fclinear(out)
        return out


class ResBlock(nn.Module):
    ''' residual block'''
    def __init__(self, in_channel, out_channel, stride):
        super().__init__()
        '''
        in_channel: number of channels in the input image.
        out_channel: number of channels produced by the convolution.
        stride: stride of the convolution.
        '''
        # 1. define double convolution
             # convolution
             # batch normalization
             # activate function
             # ......
        self.conv1 = conv_block(in_channel, out_channel, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel)
        )

        # 2. if in_channel != out_channel or stride != 1, deifine 1x1 convolution layer to change the channel or size.
        if in_channel != out_channel or stride != 1:
            self.conv3 = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.conv3 = nn.Identity()
        self.relu = nn.ReLU()
        # Note: we are going to implement 'Basic residual block' by above steps, you can also implement 'Bottleneck Residual block'

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # 1. convolve the input
        # 2. if in_channel != out_channel or stride != 1, change the channel or size of 'x' using 1x1 convolution.
        # 3. Add the output of the convolution and the original data (or from 2.)
        # 4. relu
        out = self.conv1(x)
        out = self.conv2(out)
        x = self.conv3(x)
        out = self.relu(out + x)
        return out



class ResNet(nn.Module):
    '''residual network'''
    def __init__(self):
        super().__init__()

        # 1. define convolution layer to process raw RGB image
        self.prestage = nn.Sequential(
            conv_block(3, 64, kernel_size=7),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.resstage1 = nn.Sequential(
            ResBlock(64, 128, stride=1),
            ResBlock(128, 128, stride=1),
        )
        self.resstage2 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256, stride=1),
        )
        self.resstage3 = nn.Sequential(
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512, stride=1),
        )
        self.fclinear = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )
        # 2. define multiple residual blocks
        # 3. define full-connected layer to classify

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        # classification
        out = self.prestage(x)
        out = self.resstage1(out)
        out = self.resstage2(out)
        out = self.resstage3(out)
        out = self.fclinear(out)
        return out
    

class ResNextBlock(nn.Module):
    '''ResNext block'''
    def __init__(self, in_channel, out_channel, bottle_neck, group, stride):
        super().__init__()
        # in_channel: number of channels in the input image
        # out_channel: number of channels produced by the convolution
        # bottle_neck: int, bottleneck= out_channel / hidden_channel 
        # group: number of blocked connections from input channels to output channels
        # stride: stride of the convolution.

        # 1. define convolution
             # 1x1 convolution
             # batch normalization
             # activate function
             # 3x3 convolution
             # ......
             # 1x1 convolution
             # ......
        hidden_channel = out_channel // bottle_neck * group
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=hidden_channel, kernel_size=1, stride=stride),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=3, padding=1, stride=1, groups=group),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channel, out_channels=out_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channel)
        )

        # 2. if in_channel != out_channel or stride != 1, deifine 1x1 convolution layer to change the channel or size.
        if in_channel != out_channel or stride != 1:
            self.conv4 = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.conv4 = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # 1. convolve the input
        # 2. if in_channel != out_channel or stride != 1, change the channel or size of 'x' using 1x1 convolution.
        # 3. Add the output of the convolution and the original data (or from 2.)
        # 4. relu
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        x = self.conv4(x)
        out = self.relu(out + x)
        return out


class ResNext(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. define convolution layer to process raw RGB image
        # 2. define multiple residual blocks
        # 3. define full-connected layer to classify
        self.prestage = nn.Sequential(
            conv_block(3, 64, kernel_size=7),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.resstage1 = nn.Sequential(
            ResNextBlock(64, 128, bottle_neck=4, group=4, stride=1),
            ResNextBlock(128, 128, bottle_neck=4, group=4, stride=1),
        )
        self.resstage2 = nn.Sequential(
            ResNextBlock(128, 256, bottle_neck=4, group=4, stride=2),
            ResNextBlock(256, 256, bottle_neck=4, group=4, stride=1),
        )
        self.resstage3 = nn.Sequential(
            ResNextBlock(256, 512, bottle_neck=4, group=4, stride=2),
            ResNextBlock(512, 512, bottle_neck=4, group=4, stride=1),
        )
        self.fclinear = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        # classification
        out = self.prestage(x)
        out = self.resstage1(out)
        out = self.resstage2(out)
        out = self.resstage3(out)
        out = self.fclinear(out)
        return out

