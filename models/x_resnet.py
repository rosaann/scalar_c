#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:24:47 2019

@author: zl
"""

import torch.nn as nn
import torch
#from .utils import load_state_dict_from_url


__all__ = ['XResNet', 'Xresnet18', 'Xresnet34', 'Xresnet50', 'Xresnet101',
           'Xresnet152', 'Xresnext50_32x4d', 'Xresnext101_32x8d']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


def conv3x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv2x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(2, 1), stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        print('x ', x.shape)
        identity = x

        out = self.conv1(x)
        print('x1 ', out.shape)
        out = self.bn1(out)
        print('x2 ', out.shape)
        out = self.relu(out)
        print('x3 ', out.shape)
        out = self.conv2(out)
        print('x4 ', out.shape)
        out = self.bn2(out)
        print('x5 ', out.shape)
        if self.downsample is not None:
            identity = self.downsample(x)
            print('identity ', identity.shape)
        out += identity
        
        print('x6 ', out.shape)
        out = self.relu(out)
        print('x7 ', out.shape)
        return out
class BasicBlock_1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock_1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        
        print('b inplanes ', inplanes, ' planes ', planes, ' stride ', stride)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=(3, 1), stride=1,
                     padding=0, groups=1, bias=False, dilation=0)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 1), stride=1,
                     padding=0, groups=1, bias=False, dilation=0)
        self.bn2 = norm_layer(64)
        self.downsample = nn.Sequential(
                
                nn.Conv2d(33, 64, kernel_size=1, stride=1, bias=False, padding = 2),
                norm_layer(64),
            )
        self.stride = stride

    def forward(self, x):
        print('x ', x.shape)
        identity = x

        out = self.conv1(x)
        print('x1 ', out.shape)
        out = self.bn1(out)
        print('x2 ', out.shape)
        out = self.relu(out)
        print('x3 ', out.shape)
        out = self.conv2(out)
        print('x4 ', out.shape)
        out = self.bn2(out)
        print('x5 ', out.shape)
        if self.downsample is not None:
            identity = self.downsample(x)
            print('identity ', identity.shape)
        out += identity
        
        print('x6 ', out.shape)
        out = self.relu(out)
        print('x7 ', out.shape)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x1(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class XResNet_old(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(XResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(33, self.inplanes, kernel_size=(2, 1), stride=1, padding=1,
                               )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.layer1 = self._make_layer(block, 64, layers[0])
      #  self.layer1 = BasicBlock_1(64, layers[0])
       # if stride != 1 or self.inplanes != planes * block.expansion:
        stride = 1
       

        add_layers = []
        print ('layers ', layers)
        add_layers.append(BasicBlock_1(self.inplanes, layers[0], stride, self.groups,
                            self.base_width, self.dilation, norm_layer))
        self.inplanes = layers[0] * block.expansion
        for _ in range(1, layers[0]):
            add_layers.append(BasicBlock_1(self.inplanes, 64, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        self.layer1 = nn.Sequential(*add_layers)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        print('b--x ', x.shape)
        x = self.conv1(x)
        print('b--x1 ', x.shape)
        x = self.bn1(x)
        x = self.relu(x)
      #  x = self.maxpool(x)

        x = self.layer1(x)
        print('b--x2 ', x.shape)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _xresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = XResNet(block, layers, **kwargs)
  #  if pretrained:
    #    state_dict = load_state_dict_from_url(model_urls[arch],
    #                                          progress=progress)
    #    model.load_state_dict(state_dict)
    return model


def x_resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _xresnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def x_resnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _xresnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def x_resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _xresnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def x_resnet101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _xresnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def x_resnet152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _xresnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def x_resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-50 32x4d model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _xresnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def x_resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _xresnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
pretrained, progress, **kwargs)
    
    
    
    
    
class XResNet(nn.Module):
    
    def __init__(self):
        super(XResNet, self).__init__()
        self.layers11 = nn.Sequential(nn.Conv2d(33, 64, (2,1), 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
        self.layers12 = nn.Sequential(nn.Conv2d(64, 64 ,(2,1), 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
       
        self.layers10 = nn.Sequential(nn.Conv2d(33, 64, (1,1), 1, (1, 1)),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
    #    self.lineLayer10q = nn.Linear(5952, 6336)
        #self.lineLayer10 = nn.Linear(35840, 1600)
        ###
        self.layers11s1 = nn.Sequential(nn.Conv2d(64, 64, (2,1), 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
        self.layers12s1 = nn.Sequential(nn.Conv2d(64, 64 ,(2,1), 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
       
        self.layers10s1 = nn.Sequential(nn.Conv2d(64, 64, (1,1), 1, (1, 1)),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
    #    self.lineLayer10qs1 = nn.Linear(11200, 11840)
  
        #######
        self.layers21 = nn.Sequential(nn.Conv2d(64, 96, (2,1), 1, 1),
                                nn.BatchNorm2d(96),
                                nn.ReLU(inplace=True),
                                )
        self.layers22 = nn.Sequential(nn.Conv2d(96, 96, (2,1), 1, 1),
                                nn.BatchNorm2d(96),
                                nn.ReLU(inplace=True),
                                )
       
        self.layers20 = nn.Sequential(nn.Conv2d(64, 96, (1,1), 1, 1),
                                nn.BatchNorm2d(96),
                                nn.ReLU(inplace=True),
                                )
    #    self.lineLayer20q = nn.Linear(26208, 27552)
        #########
        
        self.layers21s1 = nn.Sequential(nn.Conv2d(96, 96, (2,1), 1, 1),
                                nn.BatchNorm2d(96),
                                nn.ReLU(inplace=True),
                                )
        self.layers22s1 = nn.Sequential(nn.Conv2d(96, 96, (2,1), 1, 1),
                                nn.BatchNorm2d(96),
                                nn.ReLU(inplace=True),
                                )
       
        self.layers20s1 = nn.Sequential(nn.Conv2d(96, 96, (1,1), 1, 1),
                                nn.BatchNorm2d(96),
                                nn.ReLU(inplace=True),
                                )
     #   self.lineLayer20qs1 = nn.Linear(37152, 38880)
        #######
       
        self.layers31 = nn.Sequential(nn.Conv2d(96, 128, (2,1), 1, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                )
        self.layers32 = nn.Sequential(nn.Conv2d(128, 128, (2,1), 1, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                )
       
        self.layers30 = nn.Sequential(nn.Conv2d(96, 128, (1,1), 1, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                )
       # self.lineLayer30q = nn.Linear(66176, 68992)
        
       ###
        self.lineLayer_end = nn.Linear(68992, 841)
        
        
      #  self.layers_y1 = nn.Sequential(nn.Conv2d(1, 4, (2,1), 1, 1),
      #                          nn.BatchNorm2d(4),
     #                           nn.ReLU(inplace=True),
     #   self.layers_y2 = nn.Sequential(nn.Conv2d(4, 8, (2,1), 1, 1),
     #                           nn.BatchNorm2d(8),
     #                           nn.ReLU(inplace=True),
    #                            )
     #   self.layers_y3 = nn.Sequential(nn.Conv2d(8, 16, (2,1), 1, 1),
    #                            nn.BatchNorm2d(16),
     #                           nn.ReLU(inplace=True),
     #                           )
        
      #  self.lineLayer = nn.Linear(31648, 29*29)
    def forward(self, x):
        # Initial block
     #   print('x ', x.shape)
        x11 = self.layers11(x) 
     #   print('x11 ', x11.shape)
        x11 = self.layers12(x11)
    #    print('x12 ', x11.shape)
       # x11 = self.layers13(x11)
       # print('x13 ', x11.shape)
        x10 = self.layers10(x)
     #   print('x10 ', x10.shape)
        sha = x10.shape
        ze = torch.zeros(sha[0],sha[1], sha[2], 2) 
        x10 = torch.cat((x10, ze.cuda()), 3)
     #   print('x10. ', x10.shape)
        
        x1 = x11 + x10
    #    print('x1 ', x1.shape)
###########
        x11 = self.layers11s1(x1) 
    #    print('x11s1 ', x11.shape)
        x11 = self.layers12s1(x11)
    #    print('x12s1 ', x11.shape)
       # x11 = self.layers13(x11)
       # print('x13 ', x11.shape)
        x10 = self.layers10s1(x1)
    #    print('x10s1 ', x10.shape)
        sha = x10.shape
        ze = torch.zeros(sha[0],sha[1], sha[2], 2) 
        x10 = torch.cat((x10, ze.cuda()), 3)
     #   print('x10s1 1 ', x10.shape)
        
        x1 = x11 + x10
    #    print('x1 ', x1.shape)
  ####      
        x21 = self.layers21(x1) 
    #    print('x21 ', x21.shape)
        x21 = self.layers22(x21)
    #    print('x22 ', x21.shape)
        x20 = self.layers20(x1)
    #    print('x20 ', x20.shape)
        sha = x20.shape
        ze = torch.zeros(sha[0],sha[1], sha[2], 2) 
        x20 = torch.cat((x20, ze.cuda()), 3)
  #      print('x20 1 ', x20.shape)
        
        x2 = x21 + x20
    #    print('x2 ', x2.shape)
        
    ####
        x21 = self.layers21s1(x2) 
    #    print('x21s1 ', x21.shape)
        x21 = self.layers22s1(x21)
     #   print('x22s1 ', x21.shape)
        x20 = self.layers20s1(x2)
     #   print('x20s1 ', x20.shape)
        sha = x20.shape
        ze = torch.zeros(sha[0],sha[1], sha[2], 2) 
        x20 = torch.cat((x20, ze.cuda()), 3)
     #   print('x20s1 1 ', x20.shape)
        
        x2 = x21 + x20
    #    print('x2s1 ', x2.shape)
        
        #######
        x31 = self.layers31(x2) 
    #    print('x31 ', x31.shape)
        x31 = self.layers32(x31)
    #    print('x32 ', x31.shape)
        x30 = self.layers30(x2)
    #    print('x30 ', x30.shape)
        sha = x30.shape
        ze = torch.zeros(sha[0],sha[1], sha[2], 2) 
        x30 = torch.cat((x30, ze.cuda()), 3)
      #  
        x3 = x31 + x30
       # print('x3 ', x3.shape)
        
        x3 = x3.view(x3.shape[0], -1)
        y  = self.lineLayer_end(x3)
   #     print('y ', y.shape)
        
      #  print('layers1 ', y.shape)
    #    y = self.layers_y1(y)
     #   print('layers2 ', y.shape)
    #    y = self.layers_y2(y)
     #   print('layers3 ', y.shape)
    #    y = self.layers_y3(y)
      #  print('layers3 ', y.shape)
     #   y = y.view(y.shape[0], -1)
      #  print('view ', y.shape)
     #   y = self.lineLayer(y)
      #  print('line ', y.shape)
        y = y.reshape(y.shape[0], 29, 29)
        return y   

class XResNet2(nn.Module):
    
    def __init__(self):
        super(XResNet2, self).__init__()
        self.layers11 = nn.Sequential(nn.Conv2d(33, 64, (2,1), 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
        self.layers12 = nn.Sequential(nn.Conv2d(64, 64 ,(2,1), 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
       
        self.layers10 = nn.Sequential(nn.Conv2d(33, 64, (1,1), 1, (1, 1)),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
    #    self.lineLayer10q = nn.Linear(5952, 6336)
        #self.lineLayer10 = nn.Linear(35840, 1600)
        ###
        self.layers11s1 = nn.Sequential(nn.Conv2d(64, 64, (2,1), 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
        self.layers12s1 = nn.Sequential(nn.Conv2d(64, 64 ,(2,1), 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
       
        self.layers10s1 = nn.Sequential(nn.Conv2d(64, 64, (1,1), 1, (1, 1)),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
    #    self.lineLayer10qs1 = nn.Linear(11200, 11840)
  
        #######
        self.layers21 = nn.Sequential(nn.Conv2d(64, 128, (2,1), 1, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                )
        self.layers22 = nn.Sequential(nn.Conv2d(128, 128, (2,1), 1, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                )
       
        self.layers20 = nn.Sequential(nn.Conv2d(64, 128, (1,1), 1, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                )
    #    self.lineLayer20q = nn.Linear(26208, 27552)
        #########
        
        self.layers21s1 = nn.Sequential(nn.Conv2d(128, 128, (2,1), 1, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                )
        self.layers22s1 = nn.Sequential(nn.Conv2d(128, 128, (2,1), 1, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                )
       
        self.layers20s1 = nn.Sequential(nn.Conv2d(128, 128, (1,1), 1, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                )
     #   self.lineLayer20qs1 = nn.Linear(37152, 38880)
        #######
       
        self.layers31 = nn.Sequential(nn.Conv2d(128, 256, (2,1), 1, 1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True),
                                )
        self.layers32 = nn.Sequential(nn.Conv2d(256, 256, (2,1), 1, 1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True),
                                )
       
        self.layers30 = nn.Sequential(nn.Conv2d(128, 256, (1,1), 1, 1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True),
                                )
       # self.lineLayer30q = nn.Linear(66176, 68992)
        
       ###
        self.lineLayer_end = nn.Linear(68992, 841)
        
        
      #  self.layers_y1 = nn.Sequential(nn.Conv2d(1, 4, (2,1), 1, 1),
      #                          nn.BatchNorm2d(4),
     #                           nn.ReLU(inplace=True),
     #   self.layers_y2 = nn.Sequential(nn.Conv2d(4, 8, (2,1), 1, 1),
     #                           nn.BatchNorm2d(8),
     #                           nn.ReLU(inplace=True),
    #                            )
     #   self.layers_y3 = nn.Sequential(nn.Conv2d(8, 16, (2,1), 1, 1),
    #                            nn.BatchNorm2d(16),
     #                           nn.ReLU(inplace=True),
     #                           )
        
      #  self.lineLayer = nn.Linear(31648, 29*29)
    def forward(self, x):
        # Initial block
     #   print('x ', x.shape)
        x11 = self.layers11(x) 
     #   print('x11 ', x11.shape)
        x11 = self.layers12(x11)
    #    print('x12 ', x11.shape)
       # x11 = self.layers13(x11)
       # print('x13 ', x11.shape)
        x10 = self.layers10(x)
     #   print('x10 ', x10.shape)
        sha = x10.shape
        ze = torch.zeros(sha[0],sha[1], sha[2], 2) 
        x10 = torch.cat((x10, ze.cuda()), 3)
     #   print('x10. ', x10.shape)
        
        x1 = x11 + x10
    #    print('x1 ', x1.shape)
###########
        x11 = self.layers11s1(x1) 
    #    print('x11s1 ', x11.shape)
        x11 = self.layers12s1(x11)
    #    print('x12s1 ', x11.shape)
       # x11 = self.layers13(x11)
       # print('x13 ', x11.shape)
        x10 = self.layers10s1(x1)
    #    print('x10s1 ', x10.shape)
        sha = x10.shape
        ze = torch.zeros(sha[0],sha[1], sha[2], 2) 
        x10 = torch.cat((x10, ze.cuda()), 3)
     #   print('x10s1 1 ', x10.shape)
        
        x1 = x11 + x10
    #    print('x1 ', x1.shape)
  ####      
        x21 = self.layers21(x1) 
    #    print('x21 ', x21.shape)
        x21 = self.layers22(x21)
    #    print('x22 ', x21.shape)
        x20 = self.layers20(x1)
    #    print('x20 ', x20.shape)
        sha = x20.shape
        ze = torch.zeros(sha[0],sha[1], sha[2], 2) 
        x20 = torch.cat((x20, ze.cuda()), 3)
  #      print('x20 1 ', x20.shape)
        
        x2 = x21 + x20
    #    print('x2 ', x2.shape)
        
    ####
        x21 = self.layers21s1(x2) 
    #    print('x21s1 ', x21.shape)
        x21 = self.layers22s1(x21)
     #   print('x22s1 ', x21.shape)
        x20 = self.layers20s1(x2)
     #   print('x20s1 ', x20.shape)
        sha = x20.shape
        ze = torch.zeros(sha[0],sha[1], sha[2], 2) 
        x20 = torch.cat((x20, ze.cuda()), 3)
     #   print('x20s1 1 ', x20.shape)
        
        x2 = x21 + x20
    #    print('x2s1 ', x2.shape)
        
        #######
        x31 = self.layers31(x2) 
    #    print('x31 ', x31.shape)
        x31 = self.layers32(x31)
    #    print('x32 ', x31.shape)
        x30 = self.layers30(x2)
    #    print('x30 ', x30.shape)
        sha = x30.shape
        ze = torch.zeros(sha[0],sha[1], sha[2], 2) 
        x30 = torch.cat((x30, ze.cuda()), 3)
      #  
        x3 = x31 + x30
       # print('x3 ', x3.shape)
        
        x3 = x3.view(x3.shape[0], -1)
        y  = self.lineLayer_end(x3)
   #     print('y ', y.shape)
        
      #  print('layers1 ', y.shape)
    #    y = self.layers_y1(y)
     #   print('layers2 ', y.shape)
    #    y = self.layers_y2(y)
     #   print('layers3 ', y.shape)
    #    y = self.layers_y3(y)
      #  print('layers3 ', y.shape)
     #   y = y.view(y.shape[0], -1)
      #  print('view ', y.shape)
     #   y = self.lineLayer(y)
      #  print('line ', y.shape)
        y = y.reshape(y.shape[0], 29, 29)
        return y             