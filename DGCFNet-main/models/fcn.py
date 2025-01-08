import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F


class FCN_res18(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=False):
        super(FCN_res18, self).__init__()
        resnet = models.resnet18(pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels > 3:
            newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels - 3, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

class FCN(nn.Module):
     def __init__(self, num_classes=6,):
          super().__init__()
          self.res18 = FCN_res18(in_channels=3, num_classes=7, pretrained=False)
          self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(128, momentum=0.95),
                                    nn.ReLU())
          self.classifier = nn.Sequential(
               nn.Conv2d(128, 128, kernel_size=1),
               nn.BatchNorm2d(128, momentum=0.95),
               nn.ReLU(),
               nn.Conv2d(128, num_classes, kernel_size=1)
               )

     def forward(self, x):
          x_size = x.size()
          res0 = self.res18.layer0(x)  # 1/2, 64
          res0 = self.res18.maxpool(res0)  # 1/4, 64
          res1 = self.res18.layer1(res0)  # 1/4, 256
          res2 = self.res18.layer2(res1)  # 1/8, 512
          res3 = self.res18.layer3(res2)  # 1/8, 1024
          res4 = self.res18.layer4(res3)  # 1/8, 2048
          x = self.head(res4)
          x = self.classifier(x)

          return F.upsample(x, x_size[2:], mode='bilinear')