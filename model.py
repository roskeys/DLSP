import torch
import torch.nn as nn

class BottleNeck(nn.Module):

    # Global expansion coefficient
    expansion = 4

    def __init__(self, in_channels, out_channels, stride = 1, shortcut = False):
        """
        in_channels: input channels
        out_channels: output channels for second conv2d layer. 
            The true output channel fed to the next module will have out_channels * expansion channels
        stride: stride number
        shortcut: if input size != output size
        """
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size = 1, stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2D(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2D(out_channels, out_channels * self.expansion, kernel_size = 1, stride = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace = True)
        self.shortcut = shortcut

        if shortcut:
            self.convs = nn.Conv2D(in_channels, out_channels * self.expansion, kernel_size = 3, stride = stride, padding = 1, bias = False)
            self.bns = nn.BatchNorm2d(out_channels * self.expansion)
    
    def forward(self, model):
        identity = model

        out = self.conv1(model)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut:
            identity = self.convs(model)
            identity = self.bns(identity)
        
        out += identity
        out = self.relu(out)
        return out

class BottleNeckLayer(nn.module):

    def __init__(self, in_channels, out_channels, modules = 3, downsample = False):
        """
        in_channels: input channels
        out_channels: output channels for second conv2d layer. 
            The true output channel fed to the next module will have out_channels * expansion channels
        modules: module number, basically how many blocks in one layer
        downsample: if set to true, the first module will be set to use downsampling, causing stride = 2
        """
        super(BottleNeck, self).__init__()
        self.blocks = []
        if downsample:
            self.bottleneck1 = BottleNeck(in_channels, out_channels, stride = 2, shortcut = True)
        else:
            self.bottleneck1 = BottleNeck(in_channels, out_channels, stride = 1, shortcut = False)
        for index in range(modules - 1):
            self.blocks.append(BottleNeck(out_channels, out_channels, stride = 1, shortcut = False))        

    def forward(self, model):
        out = self.bottleneck1(model)
        for block in self.blocks:
            out = block(out)
        return out
