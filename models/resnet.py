import torch
import torch.nn as nn


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

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.branch1 = nn.Sequential(branchBasicBlock(64 * block.expansion, 128 * block.expansion, stride=2),
                                     branchBasicBlock(128 * block.expansion, 256 * block.expansion, stride=2),
                                     branchBasicBlock(256 * block.expansion, 512 * block.expansion, stride=2))
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_fc1 = nn.Linear(512 * block.expansion, num_classes)


        self.branch2 = nn.Sequential(branchBasicBlock(128 * block.expansion, 256 * block.expansion, stride=2),
                                     branchBasicBlock(256 * block.expansion, 512 * block.expansion, stride=2))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_fc2 = nn.Linear(512 * block.expansion, num_classes)

        self.branch3 = branchBasicBlock(256 * block.expansion, 512 * block.expansion, stride=2)
        self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_fc3 = nn.Linear(512 * block.expansion, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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

        x = self.layer1(x)
        middle_output1 = self.branch1(x)
        middle_output1 = self.avgpool1(middle_output1)
        middle1_fea = middle_output1
        middle_output1 = torch.flatten(middle_output1, 1)
        middle_output1 = self.middle_fc1(middle_output1)

        x = self.layer2(x)
        middle_output2 = self.branch2(x)
        middle_output2 = self.avgpool2(middle_output2)
        middle2_fea = middle_output2
        middle_output2 = torch.flatten(middle_output2, 1)
        middle_output2 = self.middle_fc2(middle_output2)

        x = self.layer3(x)
        middle_output3 = self.branch3(x)
        middle_output3 = self.avgpool3(middle_output3)
        middle3_fea = middle_output3
        middle_output3 = torch.flatten(middle_output3, 1)
        middle_output3 = self.middle_fc3(middle_output3)

        x = self.layer4(x)
        x = self.avgpool(x)
        final_fea = x
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return middle_output1, middle_output2, middle_output3, x, middle1_fea, middle2_fea, middle3_fea, final_fea


def multi_resnet50_kd(num_classes=1000):
    return Multi_ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes=num_classes)

def multi_resnet34_kd(num_classes=1000):
    return Multi_ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def multi_resnet18_kd(num_classes=1000):
    return Multi_ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

if __name__ == '__main__':
    net = multi_resnet18_kd(100)
    x = torch.rand(1, 3, 32, 32)
    l, l1, l2, l3, f, f1, f2, f3 = net(x)
    print(l.shape, l1.shape, l2.shape, l3.shape)
    print(f.shape, f1.shape, f2.shape, f3.shape)