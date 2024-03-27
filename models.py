from __future__ import division
import torch
import torch.nn as nn
import numpy as np
from math import sqrt
import torch.nn.functional as F
    
    
class localization_UNet(nn.Module):
    def __init__(self):
        super(localization_UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        def upconv_block(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # Downsampling
        self.conv1 = conv_block(1, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bridge
        self.conv4 = conv_block(256, 512)
        self.conv5 = conv_block(512, 512)

        # Upsampling
        self.upconv6 = upconv_block(512, 256)
        self.conv6 = conv_block(512, 256)
        self.upconv7 = upconv_block(256, 128)
        self.conv7 = conv_block(256, 128)
        self.upconv8 = upconv_block(128, 64)
        self.conv8 = conv_block(128, 64)

        self.output = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        x4 = self.pool2(x3)
        x5 = self.conv3(x4)
        x6 = self.pool3(x5)

        x7 = self.conv4(x6)
        x8 = self.conv5(x7)

        x9 = self.upconv6(x8)
        x10 = torch.cat([x9, x5], dim=1)
        x11 = self.conv6(x10)

        x12 = self.upconv7(x11)
        x13 = torch.cat([x12, x3], dim=1)
        x14 = self.conv7(x13)

        x15 = self.upconv8(x14)
        x16 = torch.cat([x15, x1], dim=1)
        x17 = self.conv8(x16)

        out = self.output(x17)

        return out
    

class segmentation_UNet(nn.Module):
    def __init__(self):
        super(segmentation_UNet, self).__init__()

        def conv_block(in_channels, out_channels, dropout_rate=0.5):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
        )

        def upconv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels)
        )

        # Downsampling
        self.conv1 = conv_block(1, 128)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = conv_block(128, 256)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv3 = conv_block(256, 512)
        self.pool3 = nn.AvgPool2d(2, 2)
        self.conv4 = conv_block(512, 1024)
        self.pool4 = nn.AvgPool2d(2, 2)

        # Bridge
        self.conv5 = conv_block(1024, 2048)
        self.conv6 = conv_block(2048, 2048)

        # Upsampling
        self.upconv7 = upconv_block(2048, 1024)
        self.conv7 = conv_block(2048, 1024)
        self.dropout7 = nn.Dropout(0.3)
        self.upconv8 = upconv_block(1024, 512)
        self.conv8 = conv_block(1024, 512)
        self.dropout8 = nn.Dropout(0.3)
        self.upconv9 = upconv_block(512, 256)
        self.conv9 = conv_block(512, 256)
        self.dropout9 = nn.Dropout(0.3)
        self.upconv10 = upconv_block(256, 128)
        self.conv10 = conv_block(256, 128)

        self.output = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, img1):
        #x = torch.cat((img1, img2), dim=1)
        x = img1
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        x4 = self.pool2(x3)
        x5 = self.conv3(x4)
        x6 = self.pool3(x5)
        x7 = self.conv4(x6)
        x8 = self.pool4(x7)

        x9 = self.conv5(x8)
        x10 = self.conv6(x9)

        x11 = self.upconv7(x10)
        x12 = torch.cat([x11, x7], dim=1)
        x13 = self.conv7(x12)

        x14 = self.upconv8(x13)
        x15 = torch.cat([x14, x5], dim=1)
        x16 = self.conv8(x15)

        x17 = self.upconv9(x16)
        x18 = torch.cat([x17, x3], dim=1)
        x19 = self.conv9(x18)

        x20 = self.upconv10(x19)
        x21 = torch.cat([x20, x1], dim=1)
        x22 = self.conv10(x21)

        out = self.output(x22)

        return out
    

class AtomSegNet(nn.Module):
    # class UNet(torch.jit.ScriptModule):
    def __init__(self, colordim=1):
        super(AtomSegNet, self).__init__()
        self.conv1_1 = nn.Conv2d(colordim, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.upconv4 = nn.Conv2d(256, 128, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.bn4_out = nn.BatchNorm2d(256)

        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.upconv7 = nn.Conv2d(128, 64, 1)
        self.bn7 = nn.BatchNorm2d(64)
        self.bn7_1 = nn.BatchNorm2d(128)
        self.bn7_2 = nn.BatchNorm2d(128)
        self.bn7_out = nn.BatchNorm2d(128)

        self.conv9_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv9_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn9_1 = nn.BatchNorm2d(64)
        self.bn9_2 = nn.BatchNorm2d(64)
        self.conv9_3 = nn.Conv2d(64, colordim, 1)
        self.bn9_3 = nn.BatchNorm2d(colordim)
        self.bn9 = nn.BatchNorm2d(colordim)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        #self._initialize_weights()

        # self.input_layer = nn.Sequential(self.conv1_1, self.bn1_1, nn.ReLU(),self.conv1_2, self.bn1_2,  nn.ReLU())
        #
        # self.down1 = nn.Sequential(self.conv2_1, self.bn2_1, nn.ReLU(), self.conv2_2, self.bn2_2, nn.ReLU())
        #
        # self.down2 = nn.Sequential(self.conv4_1, self.bn4_1, nn.ReLU(), self.conv4_2, self.bn4_2, nn.ReLU())
        #
        # self.up1 = nn.Sequential(self.upconv4, self.bn4)
        #
        # self.up2 = nn.Sequential(self.bn4_out, self.conv7_1,self.bn7_1 , nn.ReLU(), self.conv7_2, self.bn7_2, nn.ReLU())
        #
        # self.output = nn.Sequential(self.conv4_1, self.bn4_1, nn.ReLU(), self.conv4_2, self.bn4_2, nn.ReLU())


    def forward(self, x1):
        x1 = F.relu(self.bn1_2(self.conv1_2(F.relu(self.bn1_1(self.conv1_1(x1))))))
        x2 = F.relu(self.bn2_2(self.conv2_2(F.relu(self.bn2_1(self.conv2_1(self.maxpool(x1)))))))
        xup = F.relu(self.bn4_2(self.conv4_2(F.relu(self.bn4_1(self.conv4_1(self.maxpool(x2)))))))
        xup = self.bn4(self.upconv4(self.upsample(xup)))
        xup = self.bn4_out(torch.cat((x2, xup), 1))
        xup = F.relu(self.bn7_2(self.conv7_2(F.relu(self.bn7_1(self.conv7_1(xup))))))

        xup = self.bn7(self.upconv7(self.upsample(xup)))
        xup = self.bn7_out(torch.cat((x1, xup), 1))

        xup = F.relu(self.conv9_3(F.relu(self.bn9_2(self.conv9_2(F.relu(self.bn9_1(self.conv9_1(xup))))))))

        return torch.sigmoid(self.bn9(xup))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()