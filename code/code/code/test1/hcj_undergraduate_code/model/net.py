'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors : now more
Description : 
@LastEditTime: 2019-11-02 16:48:45
'''
from importer import *
from model.resnet import *

    
class ASSP(nn.Module):
    def __init__(self,in_channels,out_channels = 256):
        super(ASSP,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels,out_channels,1,padding = 0,dilation=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = 3,stride=1,padding = 6,dilation = 6,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = 3,stride=1,padding = 12,dilation = 12,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = 3,stride=1,padding = 18,dilation = 18,bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.conv5 = nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = 1,stride=1,padding = 0,dilation=1,bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.convf = nn.Conv2d(in_channels = out_channels * 5,out_channels = out_channels,kernel_size = 1,stride=1,padding = 0,dilation=1,bias=False)
        self.bnf = nn.BatchNorm2d(out_channels)
        self.adapool = nn.AdaptiveAvgPool2d(1)
    def forward(self,x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        x4 = self.conv4(x)
        x4 = self.bn4(x4)
        x4 = self.relu(x4)
        x5 = self.adapool(x)
        x5 = self.conv5(x5)
        x5 = self.bn5(x5)
        x5 = self.relu(x5)
        x5 = F.interpolate(x5, size = tuple(x4.shape[-2:]), mode='bilinear')
    
        x = torch.cat((x1,x2,x3,x4,x5), dim = 1) #channels first
        x = self.convf(x)
        x = self.bnf(x)
        x = self.relu(x)
        return x

# DeepLabv3+网络
class deeplabv3plus_resnet101(nn.Module):
    def __init__(self,nc=5):
        super(deeplabv3plus_resnet101,self).__init__()
        self.nc = nc
        self.backbone = ResNet_101()
        self.assp1 = ASSP(in_channels=1024)
        self.out1 = nn.Sequential(nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1,stride=1),nn.ReLU())
        self.dropout1 = nn.Dropout(0.5)
        self.up4 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=2)

        self.conv1x1 = nn.Sequential(nn.Conv2d(1024,256,1,bias=False),nn.ReLU())
        self.conv3x3 = nn.Sequential(nn.Conv2d(512,self.nc,1),nn.ReLU())
        self.dec_conv = nn.Sequential(nn.Conv2d(256,256,3,padding=1),nn.ReLU())
        
    def forward(self,x):
        x = self.backbone(x)
        out1 = self.assp1(x)
        out1 = self.out1(out1)
        out1 = self.dropout1(out1)
        out1 = self.up4(out1)

        dec = self.conv1x1(x)
        dec = self.dec_conv(dec)
        dec = self.up4(dec)
        contact = torch.cat((out1,dec),dim=1)
        out = self.conv3x3(contact)
        out = self.up4(out)
        return out
