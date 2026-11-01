import torch
import torch.nn as nn
import torch.nn.functional as F
from SLWT import LiftingScheme2D,InverseLiftingScheme2D
from DPFA import DPFA
from thop import profile

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class SGFR_Net(nn.Module):


    def __init__(self, n_channels=3, n_classes=1, deep_supervision=False):
        super(SGFR_Net, self).__init__()

        n1 = 64
        self.deep_supervision = deep_supervision
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(n_channels, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])
        self.mask = conv_block(filters[4], 1)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])
        self.Conv = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)
        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[1], n_classes, kernel_size=1, stride=1, padding=0)
            self.final2 = nn.Conv2d(filters[2], n_classes, kernel_size=1, stride=1, padding=0)
            self.final3 = nn.Conv2d(filters[3], n_classes, kernel_size=1, stride=1, padding=0)
            self.final4 = nn.Conv2d(filters[4], n_classes, kernel_size=1, stride=1, padding=0)

        self.MPU_WD = LiftingScheme2D(1,4)
        self.SGFM = InverseLiftingScheme2D(1, 4)

        self.hdpa = DPFA(filters[4], 4)

        self.mask_upsample = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        )
    def forward(self, x):
        e1 = self.Conv1(x)
        LL, LH, HL, HH = self.MPU_WD(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        e5 = self.hdpa(e5)
        mask = self.mask(e5)
        mask = self.mask_upsample(mask)
        #
        LL = LL * mask
        LH = LH * mask
        HL = HL * mask
        HH = HH * mask

        y = self.SGFM(LL, LH, HL, HH)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d2 = d2 + y
        if self.deep_supervision:
            out0 = self.Conv(d2)
            out1 = F.interpolate(self.final1(d3), x.shape[2:], mode='bilinear')
            out2 = F.interpolate(self.final2(d4), x.shape[2:], mode='bilinear')
            out3 = F.interpolate(self.final3(d5), x.shape[2:], mode='bilinear')
            out4 = F.interpolate(self.final4(e5), x.shape[2:], mode='bilinear')
            return [out0, out1, out2, out3, out4]

        else:
            out = self.Conv(d2)
            return out

if __name__ == '__main__':
    device = torch.device('cuda')
    model = SGFR_Net(1, 3).to(device)
    x = torch.randn(8, 1, 512, 512).to(device)
    out = model(x).to(device)
    print(f"out shape:{out.shape}")
    Flops, params = profile(model, inputs=(x,))
    print("Flops:%.4fG" % (Flops / 1000000000))
    print('params:% .4fM' % (params / 1000000))
