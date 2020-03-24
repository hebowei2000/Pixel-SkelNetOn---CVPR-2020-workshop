import math

import torch.nn as nn
from torchvision import models

from .modules import Bottleneck


class ResNet50(nn.Module):
    # ResNet with two branches
    def __init__(self):
        self.inplanes = 64
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        self.initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.conv1(x)  # 1/2
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.maxpool(out1)  # 1/4

        out2 = self.layer1(out1)
        out3 = self.layer2(out2)  # 1/8
        out4 = self.layer3(out3)  # 1/ 16
        out5 = self.layer4(out4)  # 1/32

        return out2, out3, out4, out5

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.state_dict().keys())
        self.load_state_dict(all_params)


class ResNet50_Dropout(nn.Module):
    # ResNet with two branches
    def __init__(self, rate=0.3):
        self.inplanes = 64
        super(ResNet50_Dropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        self.dropout_5 = nn.Dropout2d(rate)
        self.dropout_1 = nn.Dropout2d(rate)
        self.dropout_2 = nn.Dropout2d(rate)
        self.dropout_3 = nn.Dropout2d(rate)
        self.dropout_4 = nn.Dropout2d(rate)

        self.initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.conv1(x)  # 1/2
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.maxpool(out1)  # 1/4
        out1 = self.dropout_1(out1)

        out2 = self.layer1(out1)
        out2 = self.dropout_2(out2)
        out3 = self.layer2(out2)  # 1/8
        out3 = self.dropout_3(out3)
        out4 = self.layer3(out3)  # 1/ 16
        out4 = self.dropout_4(out4)
        out5 = self.layer4(out4)  # 1/32
        out5 = self.dropout_5(out5)

        return out2, out3, out4, out5

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.state_dict().keys())
        self.load_state_dict(all_params)


class ResNet101(nn.Module):
    # ResNet with two branches
    def __init__(self):
        # self.inplanes = 128
        self.inplanes = 64
        super(ResNet101, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 23, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # 1/2
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.conv2(x)  # 1/2
        # x = self.bn2(x)
        # x = self.relu2(x)
        # x = self.conv3(x)  # 1/2
        # x = self.bn3(x)
        # x = self.relu3(x)
        x = self.maxpool(x)  # 1/4

        x = self.layer1(x)
        x = self.layer2(x)  # 1/8
        x = self.layer3(x)  # 1/ 16
        x = self.layer4(x)  # 1/32

        return x
