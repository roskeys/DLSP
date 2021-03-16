import torch
import torch.nn as nn


class BottleNeck(nn.Module):
    # Global expansion coefficient
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, shortcut=False):
        """
        in_channels: input channels
        out_channels: output channels for second Conv2d layer. 
            The true output channel fed to the next module will have out_channels * expansion channels
        stride: stride number
        shortcut: if input size != output size
        """
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = shortcut

        if shortcut:
            self.convs = nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=3, stride=stride, padding=1,
                                   bias=False)
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


class BottleNeckLayer(nn.Module):

    def __init__(self, in_channels, out_channels, modules=3, downsample=False):
        """
        in_channels: input channels
        out_channels: output channels for second Conv2d layer. 
            The true output channel fed to the next module will have out_channels * expansion channels
        modules: module number, basically how many blocks in one layer
        downsample: if set to true, the first module will be set to use downsampling, causing stride = 2
        """
        super(BottleNeckLayer, self).__init__()
        self.blocks = []
        if downsample:
            self.bottleneck1 = BottleNeck(in_channels, out_channels, stride=2, shortcut=True)
        else:
            self.bottleneck1 = BottleNeck(in_channels, out_channels, stride=1, shortcut=False)
        for index in range(modules - 1):
            self.blocks.append(BottleNeck(out_channels, out_channels, stride=1, shortcut=False))

    def forward(self, model):
        out = self.bottleneck1(model)
        for block in self.blocks:
            out = block(out)
        return out


class Resnet50(nn.Module):
    def __init__(self, hidden_dim=1024, out_dim=2, dropout_p=0.5):
        super(Resnet50, self).__init__()
        self.bottle_neck_layer_1 = BottleNeckLayer(1, 64, 3, False)
        self.bottle_neck_layer_2 = BottleNeckLayer(256, 128, 4, True)
        self.bottle_neck_layer_3 = BottleNeckLayer(512, 256, 6, True)
        self.bottle_neck_layer_4 = BottleNeckLayer(1024, 512, 3, True)
        self.hidden = nn.Linear(2048, hidden_dim)
        self.out_dim = out_dim
        if out_dim == 2:
            out_dim = 1
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, image):
        x = self.bottle_neck_layer_1(image)
        x = self.bottle_neck_layer_2(x)
        x = self.bottle_neck_layer_3(x)
        x = self.bottle_neck_layer_4(x)
        x = x.view(-1)
        x = self.hidden(x)
        x = self.dropout(x)
        x = self.fc(x)
        if self.out_dim == 2:
            return torch.sigmoid(x)
        else:
            return torch.softmax(x, -1)


if __name__ == '__main__':
    x = torch.randn((32, 1, 150, 150))
    model = Resnet50()
    out = model(x)
