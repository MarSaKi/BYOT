import torch
import torch.nn as nn
from utils import utils

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def conv1x1(in_planes, planes, stride=1):
    return nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)


def branchBottleNeck(channel_in, channel_out, kernel_size):
    middle_channel = channel_out // 4
    return nn.Sequential(
        nn.Conv2d(channel_in, middle_channel, kernel_size=1, stride=1),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),

        nn.Conv2d(middle_channel, middle_channel, kernel_size=kernel_size, stride=kernel_size),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),

        nn.Conv2d(middle_channel, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
    )

def branchBasicBlock(channel_in, channel_out, stride):
    downsample = None
    if stride != 1:
        downsample = nn.Sequential(
            conv1x1(channel_in, channel_out, stride),
            nn.BatchNorm2d(channel_out),
        )
    return BasicBlock(channel_in, channel_out, stride=2, downsample=downsample)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)

        if self.downsample is not None:
            residual = self.downsample(x)

        output += residual
        output = self.relu(output)
        return output


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn3(output)

        if self.downsample is not None:
            residual = self.downsample(x)

        output += residual
        output = self.relu(output)

        return output

branchDict = {
    'c1': lambda block, num_classes: nn.Sequential(branchBasicBlock(64 * block.expansion, 128 * block.expansion, stride=2),
                                                   branchBasicBlock(128 * block.expansion, 256 * block.expansion, stride=2),
                                                   branchBasicBlock(256 * block.expansion, 512 * block.expansion, stride=2),
                                                   nn.AdaptiveAvgPool2d((1, 1))),
    'c2': lambda block, num_classes: nn.Sequential(branchBasicBlock(128 * block.expansion, 256 * block.expansion, stride=2),
                                                   branchBasicBlock(256 * block.expansion, 512 * block.expansion, stride=2),
                                                   nn.AdaptiveAvgPool2d((1, 1))),
    'c3': lambda block, num_classes: nn.Sequential(branchBasicBlock(256 * block.expansion, 512 * block.expansion, stride=2),
                                                   nn.AdaptiveAvgPool2d((1, 1))),
    'c4': lambda block, num_classes: nn.Sequential(nn.AdaptiveAvgPool2d((1, 1))),
    }

class Multi_ResNet(nn.Module):
    """Resnet model
    Args:
        block (class): block type, BasicBlock or BottleneckBlock
        layers (int list): layer num in each block
        num_classes (int): class num
    """

    def __init__(self, block, layers, num_classes=1000):
        super(Multi_ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList()
        cs = [64, 128, 256, 512]
        for i, layer_num in enumerate(layers):
            stride = 1 if i==0 else 2
            self.layers.append(self._make_layer(block, cs[i], layer_num))

        self.branch = branchDict['c{}'.format(len(layers))](block, num_classes)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, layers, stride=1):
        """A block with 'layers' layers
        Args:
            block (class): block type
            planes (int): output channels = planes * expansion
            layers (int): layer num in the block
            stride (int): the first layer stride in the block
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layer = []
        layer.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, layers):
            layer.append(block(self.inplanes, planes))

        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for layer in self.layers:
            x = layer(x)
        x = self.branch(x)

        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits

def c1(num_classes=1000):
    return Multi_ResNet(BasicBlock, [3], num_classes=num_classes)

def c2(num_classes=1000):
    return Multi_ResNet(BasicBlock, [3, 4], num_classes=num_classes)

def c3(num_classes=1000):
    return Multi_ResNet(BasicBlock, [3, 4, 6], num_classes=num_classes)

def c4(num_classes=1000):
    return Multi_ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

if __name__ == '__main__':
    net1 = c1(100)
    net2 = c2(100)
    net3 = c3(100)
    net4 = c4(100)
    x = torch.rand(1, 3, 32, 32)
    l1, l2, l3, l4 = net1(x), net2(x), net3(x), net4(x)
    print(utils.count_params(net1), utils.count_params(net2), utils.count_params(net3), utils.count_params(net4))
    print(l1.shape, l1.shape, l2.shape, l3.shape)