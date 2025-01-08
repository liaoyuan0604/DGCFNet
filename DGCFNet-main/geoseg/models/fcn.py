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

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )
class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8, dim=64):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.pre_conv(res) + x
        return x

class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.pre_conv(res) + x
        return x

class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.pre_conv = Conv(512, decode_channels, kernel_size=1)
        #        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)

        self.p3 = WF(256, decode_channels)

        self.p2 = WF(128, decode_channels)

        self.p1 = FeatureRefinementHead(64, decode_channels)

        self.segmentation_head = nn.Sequential(Conv(64, 64),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(64, num_classes, kernel_size=1))
        self.init_weight()

        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)

    def forward(self, res1, res2, res3, res4, h, w):

        x = self.b4(self.pre_conv(res4))
        x = self.p3(x, res3)
        x = self.b3(x)

        x = self.p2(x, res2)
        x = self.b2(x)

        x = self.p1(x, res1)

        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class FCN(nn.Module):
     def __init__(self, num_classes=6,dropout=0.1,
                  decode_channels=64,
                  window_size=8,
                  bilinear=False):
          super().__init__()

          self.res18 = FCN_res18(in_channels=3, num_classes=7, pretrained=False)
          self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(128, momentum=0.95),
                                    nn.ReLU())
          self.bilinear = bilinear  # 上采样方式
          self.classifier = nn.Sequential(
               nn.Conv2d(128, 128, kernel_size=1),
               nn.BatchNorm2d(128, momentum=0.95),
               nn.ReLU(),
               nn.Conv2d(128, num_classes, kernel_size=1)
               )

          self.decoder = Decoder(512, decode_channels, dropout, window_size, num_classes)
     def forward(self, x ):
          h, w = x.size()[-2:]
          res0 = self.res18.layer0(x)  # 1/2, 64
          res0 = self.res18.maxpool(res0)  # 1/4, 64
          res1 = self.res18.layer1(res0)  # 1/4, 256
          res2 = self.res18.layer2(res1)  # 1/8, 512
          res3 = self.res18.layer3(res2)  # 1/8, 1024
          res4 = self.res18.layer4(res3)  # 1/8, 2048

          x = self.decoder(res1, res2, res3, res4, h, w)

          return x