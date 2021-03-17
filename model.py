import torch
import torch.nn as nn
import torch.nn.functional as F


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
        print("New BottleNeck Block.", "The input shape is:", model.shape)

        out = self.conv1(model)
        out = self.bn1(out)
        out = self.relu(out)
        print("First Conv Layer output shape is:", out.shape)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        print("Second Conv Layer output shape is:", out.shape)

        out = self.conv3(out)
        out = self.bn3(out)
        print("Third Conv Layer output shape is:", out.shape)

        if self.shortcut:
            identity = self.convs(model)
            identity = self.bns(identity)
            print("Shortcut Conv Layer output shape is:", identity.shape)

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
        self.expansion = 4
        self.blocks = []
        if downsample:
            self.bottleneck1 = BottleNeck(in_channels, out_channels, stride=2, shortcut=True)
        else:
            self.bottleneck1 = BottleNeck(in_channels, out_channels, stride=1, shortcut=False)
        for index in range(modules - 1):
            self.blocks.append(BottleNeck(out_channels * self.expansion, out_channels, stride=1, shortcut=False))

    def forward(self, model):
        out = self.bottleneck1(model)
        for block_index in range(len(self.blocks)):
            out = self.blocks[block_index](out)
        return out


class Resnet50(nn.Module):
    def __init__(self, hidden_dim=1024, out_dim=2, dropout_p=0.5, device=None):
        super(Resnet50, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, padding=3, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.bottle_neck_layer_1 = BottleNeckLayer(64, 64, 3, False)
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
        x = torch.relu(self.conv(image))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.bottle_neck_layer_1(x)
        x = self.bottle_neck_layer_2(x)
        x = self.bottle_neck_layer_3(x)
        x = self.bottle_neck_layer_4(x)
        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = x.view(x.size(0), -1)
        x = self.hidden(x)
        x = self.dropout(x)
        x = self.fc(x)
        if self.out_dim == 2:
            return torch.sigmoid(x)
        else:
            return torch.softmax(x, -1)


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn((32, 1, 150, 150))
    model = Resnet50()
    out = model(x)
# class Bottleneck(nn.Module):
#     """
#     in_channels: input channels
#     out_channels: output channels for second Conv2d layer.
#         The true output channel fed to the next module will have out_channels * expansion channels
#     stride: stride number
#     down_sampling: if set to true, the first module will be set to use downsampling, causing stride = 2
#     expansion: expansion rate of channels
#     """
#
#     def __init__(self, in_channels, out_channels, stride=1, down_sampling=False, expansion=4):
#         super(Bottleneck, self).__init__()
#         self.expansion = expansion
#         self.down_sampling = down_sampling
#         self.bottle_neck = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
#                       bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1, stride=1,
#                       bias=False),
#             nn.BatchNorm2d(out_channels * self.expansion)
#         )
#         if self.down_sampling:
#             self.down_sampling = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channels, out_channels=out_channels * self.expansion, kernel_size=1,
#                           stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * self.expansion)
#             )
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         residual = x
#         out = self.bottle_neck(x)
#         if self.down_sampling:
#             residual = self.down_sampling(x)
#         out += residual
#         out = self.relu(out)
#         return out
#
#
# class ResNet(nn.Module):
#     """
#     Resnet class
#     blocks: Number of blocks for each bottle neck
#     num_classes: Output categories
#     expansion: expansion rate of channels
#     """
#
#     def __init__(self, blocks, num_classes=1, expansion=4):
#         super(ResNet, self).__init__()
#         self.expansion = expansion
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3,
#                       bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         )
#         self.bottle_neck_1 = self.make_layer(in_channels=64, out_channels=64, block=blocks[0], stride=1)
#         self.bottle_neck_2 = self.make_layer(in_channels=256, out_channels=128, block=blocks[1], stride=2)
#         self.bottle_neck_3 = self.make_layer(in_channels=512, out_channels=256, block=blocks[2], stride=2)
#         self.bottle_neck_4 = self.make_layer(in_channels=1024, out_channels=512, block=blocks[3], stride=2)
#         self.avgpool = nn.AvgPool2d(kernel_size=5, stride=1)
#         self.fc = nn.Linear(2048, num_classes)
#
#     def make_layer(self, in_channels, out_channels, block, stride):
#         layers = [Bottleneck(in_channels=in_channels, out_channels=out_channels, stride=stride, down_sampling=True)]
#         for i in range(1, block):
#             layers.append(Bottleneck(in_channels=out_channels * self.expansion, out_channels=out_channels))
#         return nn.Sequential(*layers)
#
#     def forward(self, image):
#         x = self.conv(image)
#         x = self.bottle_neck_1(x)
#         x = self.bottle_neck_2(x)
#         x = self.bottle_neck_3(x)
#         x = self.bottle_neck_4(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
#
#
# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x = torch.randn((32, 1, 150, 150)).cuda()
#     model = ResNet([3, 4, 6, 3]).cuda()
#     out = model(x)
