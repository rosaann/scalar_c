import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_planes),)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(Encoder, self).__init__()
        self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias)
        self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        return x


class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x
class CNNNet(nn.Module):
    #r2:0.56;mae:13.4
    def __init__(self):
        super(CNNNet, self).__init__()
        self.layers1 = nn.Sequential(nn.Conv2d(4, 8, (2,1), 1, 1),
                                nn.BatchNorm2d(8),
                                nn.ReLU(inplace=True),
                                )
        self.layers2 = nn.Sequential(nn.Conv2d(8, 16, (2,1), 1, 1),
                                nn.BatchNorm2d(16),
                                nn.ReLU(inplace=True),
                                )
        self.layers3 = nn.Sequential(nn.Conv2d(16, 32, (2,1), 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                )
        
        self.lineLayer = nn.Linear(4480, 29*29)
    def forward(self, x):
        # Initial block
     #   print('x ', x.shape)
        x = self.layers1(x)  
     #   print('layers1 ', x.shape)
        x = self.layers2(x)
     #   print('layers2 ', x.shape)
        x = self.layers3(x)
     #   print('layers3 ', x.shape)
        x = x.view(x.shape[0], -1)
     #   print('view ', x.shape)
        x = self.lineLayer(x)
     #   print('line ', x.shape)
        x = x.reshape(x.shape[0], 29, 29)
        return x
class NetX2(nn.Module):
    
    def __init__(self):
        super(NetX2, self).__init__()
        self.layers11 = nn.Sequential(nn.Conv2d(33, 64, (2,1), 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
        self.layers12 = nn.Sequential(nn.Conv2d(64, 128 ,(2,1), 1, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                )
        self.layers13 = nn.Sequential(nn.Conv2d(128, 256 ,(2,1), 1, 1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True),
                                )
        self.layers10 = nn.Sequential(nn.Conv2d(33, 256, (1,1), 1, 1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True),
                                )
        self.lineLayer10 = nn.Linear(35840, 1600)
        
        self.layers21 = nn.Sequential(nn.Conv2d(33, 64, (3,1), 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
        self.layers22 = nn.Sequential(nn.Conv2d(64, 128, (2,1), 1, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                )
        self.layers23 = nn.Sequential(nn.Conv2d(128, 256, (2,1), 1, 1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True),
                                )
        self.layers20 = nn.Sequential(nn.Conv2d(33, 256, (1,1), 1, 1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True),
                                )
        self.lineLayer20 = nn.Linear(8448, 1600)
        
        self.layers31 = nn.Sequential(nn.Conv2d(33, 64, (4,1), 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
        self.layers32 = nn.Sequential(nn.Conv2d(64, 128, (3,1), 1, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                )
        self.layers33 = nn.Sequential(nn.Conv2d(128, 154, (2,1), 1, 1),
                                nn.BatchNorm2d(154),
                                nn.ReLU(inplace=True),
                                )
        self.layers30 = nn.Sequential(nn.Conv2d(33, 32, (1,1), 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                )
        self.lineLayer30 = nn.Linear(4480, 4096)
        
        
        self.layers41 = nn.Sequential(nn.Conv2d(33, 64, (5,1), 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
        self.layers42 = nn.Sequential(nn.Conv2d(64, 128, (4,1), 1, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                )
        self.layers43 = nn.Sequential(nn.Conv2d(128, 154, (3,1), 1, 1),
                                nn.BatchNorm2d(154),
                                nn.ReLU(inplace=True),
                                )
        self.layers44 = nn.Sequential(nn.Conv2d(154, 164, (2,1), 1, 1),
                                nn.BatchNorm2d(164),
                                nn.ReLU(inplace=True),
                                )
        self.layers40 = nn.Sequential(nn.Conv2d(33, 64, (1,1), 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
        self.lineLayer40 = nn.Linear(4480, 4096)
        
        self.layers_y1 = nn.Sequential(nn.Conv2d(1, 4, (2,1), 1, 1),
                                nn.BatchNorm2d(4),
                                nn.ReLU(inplace=True),
                                )
        self.layers_y2 = nn.Sequential(nn.Conv2d(4, 8, (2,1), 1, 1),
                                nn.BatchNorm2d(8),
                                nn.ReLU(inplace=True),
                                )
        self.layers_y3 = nn.Sequential(nn.Conv2d(8, 16, (2,1), 1, 1),
                                nn.BatchNorm2d(16),
                                nn.ReLU(inplace=True),
                                )
        
        self.lineLayer = nn.Linear(31648, 29*29)
    def forward(self, x):
        # Initial block
        print('x ', x.shape)
        x11 = self.layers11(x) 
        print('x11 ', x11.shape)
        x11 = self.layers12(x11)
        print('x12 ', x11.shape)
        x11 = self.layers13(x11)
        print('x13 ', x11.shape)
        x10 = self.layers10(x)
        print('x10 ', x10.shape)
        x1 = x11.view(x11.shape[0], -1)
        print('view ', x1.shape)
        x1 = self.lineLayer10(x1)
        print('x1 ', x1.shape)
        
        x21 = self.layers21(x) 
        print('x21 ', x21.shape)
        x21 = self.layers22(x21)
        print('x22 ', x21.shape)
        x21 = self.layers23(x21)
        print('x23 ', x21.shape)
        x20 = self.layers20(x)
        print('x20 ', x20.shape)
        x2 = x21.view(x21.shape[0], -1)
        print('view ', x21.shape)
        x2 = self.lineLayer20(x2)
        print('x2 ', x2.shape)
        
        x31 = self.layers31(x) 
        print('x31 ', x31.shape)
        x31 = self.layers32(x31)
        print('x32 ', x31.shape)
        x31 = self.layers33(x31)
        print('x33 ', x31.shape)
      ##  print('x30 ', x30.shape)
        x3 = x31.view(x31.shape[0], -1)
        print('view ', x3.shape)
        x3 = self.lineLayer30(x3)
        print('x3 ', x3.shape)
        
        x41 = self.layers41(x) 
        print('x41 ', x41.shape)
        x41 = self.layers42(x41)
        print('x42 ', x41.shape)
        x41 = self.layers43(x41)
        print('x43 ', x41.shape)
        x41 = self.layers44(x41)
        print('x44 ', x41.shape)
        x40 = self.layers40(x)
        print('x40 ', x40.shape)
        x4 = x41.view(x41.shape[0], -1)
        print('view ', x4.shape)
        x4 = self.lineLayer40(x4)
        print('x4 ', x4.shape)
        
        y = x1 + x2 +x3 + x4
        y = y.view(x1.shape[0], 1, 40, 40)
        
      #  print('layers1 ', y.shape)
        y = self.layers_y1(y)
     #   print('layers2 ', y.shape)
        y = self.layers_y2(y)
     #   print('layers3 ', y.shape)
        y = self.layers_y3(y)
      #  print('layers3 ', y.shape)
        y = y.view(y.shape[0], -1)
      #  print('view ', y.shape)
        y = self.lineLayer(y)
      #  print('line ', y.shape)
        y = y.reshape(y.shape[0], 29, 29)
        return y        
class NetX(nn.Module):
    
    def __init__(self):
        super(NetX, self).__init__()
        self.layers11 = nn.Sequential(nn.Conv2d(33, 64, (2,1), 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
        self.layers12 = nn.Sequential(nn.Conv2d(64, 128 ,(2,1), 1, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                )
        
        self.layers10 = nn.Sequential(nn.Conv2d(33, 64, (1,1), 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
        self.lineLayer10 = nn.Linear(12672, 1600)
        
        self.layers21 = nn.Sequential(nn.Conv2d(33, 64, (3,1), 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
        self.layers22 = nn.Sequential(nn.Conv2d(64, 128, (2,1), 1, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                )
        self.layers20 = nn.Sequential(nn.Conv2d(33, 64, (1,1), 1, 1),
                                nn.BatchNorm2d(16),
                                nn.ReLU(inplace=True),
                                )
        self.lineLayer20 = nn.Linear(8448, 1600)
        
        self.layers31 = nn.Sequential(nn.Conv2d(33, 64, (4,1), 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
        self.layers32 = nn.Sequential(nn.Conv2d(64, 128, (3,1), 1, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                )
        self.layers33 = nn.Sequential(nn.Conv2d(128, 154, (2,1), 1, 1),
                                nn.BatchNorm2d(154),
                                nn.ReLU(inplace=True),
                                )
        self.layers30 = nn.Sequential(nn.Conv2d(33, 32, (1,1), 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                )
        self.lineLayer30 = nn.Linear(4480, 4096)
        
        
        self.layers41 = nn.Sequential(nn.Conv2d(33, 64, (5,1), 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
        self.layers42 = nn.Sequential(nn.Conv2d(64, 128, (4,1), 1, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                )
        self.layers43 = nn.Sequential(nn.Conv2d(128, 154, (3,1), 1, 1),
                                nn.BatchNorm2d(154),
                                nn.ReLU(inplace=True),
                                )
        self.layers44 = nn.Sequential(nn.Conv2d(154, 164, (2,1), 1, 1),
                                nn.BatchNorm2d(164),
                                nn.ReLU(inplace=True),
                                )
        self.layers40 = nn.Sequential(nn.Conv2d(33, 64, (1,1), 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                )
        self.lineLayer40 = nn.Linear(4480, 4096)
        
        self.layers_y1 = nn.Sequential(nn.Conv2d(1, 4, (2,1), 1, 1),
                                nn.BatchNorm2d(4),
                                nn.ReLU(inplace=True),
                                )
        self.layers_y2 = nn.Sequential(nn.Conv2d(4, 8, (2,1), 1, 1),
                                nn.BatchNorm2d(8),
                                nn.ReLU(inplace=True),
                                )
        self.layers_y3 = nn.Sequential(nn.Conv2d(8, 16, (2,1), 1, 1),
                                nn.BatchNorm2d(16),
                                nn.ReLU(inplace=True),
                                )
        
        self.lineLayer = nn.Linear(31648, 29*29)
    def forward(self, x):
        # Initial block
      #  print('x ', x.shape)
        x11 = self.layers11(x) 
     #   print('x11 ', x11.shape)
        x11 = self.layers12(x11)
     #   print('x12 ', x11.shape)
      #  x11 = self.layers13(x11)
      #  print('x13 ', x11.shape)
      #  x10 = self.layers10(x)
      #  print('x10 ', x10.shape)
        x1 = x11.view(x11.shape[0], -1)
      #  print('view ', x1.shape)
        x1 = self.lineLayer10(x1)
      #  print('x1 ', x1.shape)
        
        x21 = self.layers21(x) 
      #  print('x21 ', x21.shape)
        x21 = self.layers22(x21)
     #   print('x22 ', x21.shape)
      #  x20 = self.layers20(x)
      #  print('x20 ', x20.shape)
        x2 = x21.view(x21.shape[0], -1)
      #  print('view ', x21.shape)
        x2 = self.lineLayer20(x2)
      #  print('x2 ', x2.shape)
        
       # x31 = self.layers31(x) 
      #  print('x31 ', x31.shape)
      #  x31 = self.layers32(x31)
      #  print('x32 ', x31.shape)
      #  x31 = self.layers33(x31)
      #  print('x33 ', x31.shape)
      ##  print('x30 ', x30.shape)
       # x3 = x31.view(x31.shape[0], -1)
       # print('view ', x3.shape)
       # x3 = self.lineLayer30(x3)
      #  print('x3 ', x3.shape)
        
      #  x41 = self.layers41(x) 
      #  print('x41 ', x41.shape)
      #  x41 = self.layers42(x41)
      #  print('x42 ', x41.shape)
      #  x41 = self.layers43(x41)
      #  print('x43 ', x41.shape)
      #  x41 = self.layers44(x41)
      #  print('x44 ', x41.shape)
       # x40 = self.layers40(x)
       # print('x40 ', x40.shape)
      #  x4 = x41.view(x41.shape[0], -1)
      #  print('view ', x4.shape)
      #  x4 = self.lineLayer40(x4)
     #   print('x4 ', x4.shape)
        
        y = x1 + x2
        y = y.view(x1.shape[0], 1, 40, 40)
        
      #  print('layers1 ', y.shape)
        y = self.layers_y1(y)
     #   print('layers2 ', y.shape)
        y = self.layers_y2(y)
     #   print('layers3 ', y.shape)
        y = self.layers_y3(y)
      #  print('layers3 ', y.shape)
        y = y.view(y.shape[0], -1)
      #  print('view ', y.shape)
        y = self.lineLayer(y)
      #  print('line ', y.shape)
        y = y.reshape(y.shape[0], 29, 29)
        return y
        
class LinkNet(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self, n_classes=1):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(LinkNet, self).__init__()

        base = resnet.resnet18(pretrained=False)
        base.conv1 = nn.Conv2d(in_channels=4,
                            out_channels=base.conv1.out_channels,
                            kernel_size=2,
                            stride=1,
                            padding=0,
                            bias=base.conv1.bias)
        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
     #   self.encoder3 = base.layer3
     #   self.encoder4 = base.layer4

     #   self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
     #   self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(64, 32, 2, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)
        self.lsm = nn.LogSoftmax(dim=1)


    def forward(self, x):
        # Initial block
     #   print('x ', x.shape)
        x = self.in_block(x)
     #   print('x1 ', x.shape)
        # Encoder blocks
        e1 = self.encoder1(x)
     #   print('e1 ', e1.shape)
        e2 = self.encoder2(e1)
     #   print('e2 ', e2.shape)
      #  e3 = self.encoder3(e2)
      #  print('e3 ', e3.shape)
      #  e4 = self.encoder4(e3)
      #  print('e4 ', e4.shape)

        # Decoder blocks
        #d4 = e3 + self.decoder4(e4)
      #  d4 = self.decoder4(e4)
      #  print('d4 ', d4.shape)
     #   d4 = e3 + d4
     #   print('d4 2 ', d4.shape)
     #   d3 = self.decoder3(d4)
     
        
     #   print('d3 ', d3.shape)
     #   d3 = e2 + d3
     #   print('d3 2 ', d3.shape)
     #   d2 = self.decoder2(d3)
     #   print('d2 ', d2.shape)
     #   d2 = e1 + d2
     #   print('d2 2 ', d2.shape)
     #   d1 = self.decoder1(d2)
     #   print('d1 ', d1.shape)
     ##   print('d1 2 ', d1.shape)

        # Classifier
        y = self.tp_conv1(e2)
     #   print('y ', y.shape)
        y = self.conv2(y)
    #    print('y1 ', y.shape)
        y = self.tp_conv2(y)
     #   print('y2 ', y.shape)

       # y = self.lsm(y)
     #   print('y3 ', y.shape)
       # y.append(0.0)
        y = y[:,:,:-1,:-1]
      #  print('y4 ', y.shape)

        return y

class LinkNetBase(nn.Module):
    """
    Generate model architecture
    """

    def __init__(self, n_classes=1):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(LinkNetBase, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.encoder1 = Encoder(64, 64, 3, 1, 1)
        self.encoder2 = Encoder(64, 128, 3, 2, 1)
        self.encoder3 = Encoder(128, 256, 3, 2, 1)
        self.encoder4 = Encoder(256, 512, 3, 2, 1)

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Initial block
        print('x ', x.shape)
        x = self.conv1(x)
        print('x1 ', x.shape)
        x = self.bn1(x)
        print('x2 ', x.shape)
        x = self.relu(x)
        print('x3 ', x.shape)
        x = self.maxpool(x)
        print('x4 ', x.shape)

        # Encoder blocks
        e1 = self.encoder1(x)
        print('x5 ', x.shape)
        e2 = self.encoder2(e1)
        print('x6 ', x.shape)
        e3 = self.encoder3(e2)
        print('x7 ', x.shape)
        e4 = self.encoder4(e3)
        print('x8 ', x.shape)

        # Decoder blocks
        #d4 = e3 + self.decoder4(e4)
        d4 = e3 + self.decoder4(e4)
        print('d4 ', d4.shape)
        d3 = e2 + self.decoder3(d4)
        print('d3 ', d3.shape)
        d2 = e1 + self.decoder2(d3)
        print('d2 ', d2.shape)
        d1 = x + self.decoder1(d2)
        print('d1 ', d1.shape)

        # Classifier
        y = self.tp_conv1(d1)
        print('y1 ', y.shape)
        y = self.conv2(y)
        print('y2 ', y.shape)
        y = self.tp_conv2(y)
        print('y3 ', y.shape)

        y = self.lsm(y)
        print('y4 ', y.shape)

        return y
