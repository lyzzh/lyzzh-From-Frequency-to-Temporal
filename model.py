import os
import sys

current_path = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(current_path)[0]
sys.path.append(current_path)
sys.path.append(rootPath)

import torch
import torch.nn as nn
from TCN_util import TemporalConvNet
from util import Conv2dWithConstraint, LinearWithConstraint


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups, max_norm=1.):
        super(ResidualBlock, self).__init__()

        self.conv1 = Conv2dWithConstraint(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=(1, 1), groups=groups, bias=False, max_norm=max_norm)
        self.conv2 = Conv2dWithConstraint(in_channels=out_channels, out_channels=out_channels,
                                          kernel_size=(1, 3), padding='same', groups=groups, bias=False, max_norm=max_norm)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(residual)
        return out



class TCI(nn.Module):
    def __init__(self, num_classes, chans, samples=1125,
                 FT=32, dropout_rate=0.5, pooling_size=(3, 3),
                 tcn_kernelSize=4, tcn_dropout=0.3):
        super(TCI, self).__init__()

        self.stage1 = nn.Sequential(
            Conv2dWithConstraint(in_channels=1, out_channels=FT,
                                 kernel_size=(1, 64), padding='same',
                                 bias=False),

            nn.BatchNorm2d(FT),
            nn.ELU()
        )

        self.stage2 = nn.Sequential(
            Conv2dWithConstraint(in_channels=FT, out_channels=2 * FT,
                                 kernel_size=(3, 1),
                                 bias=False,
                                 groups=FT,
                                 max_norm=1.),
            nn.BatchNorm2d(2*FT),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(dropout_rate)
        )

        self.block3 = nn.Sequential(
            ResidualBlock(in_channels=2 * FT, out_channels=2 * FT, groups=FT),
        )
        self.maxpooling = nn.MaxPool2d(
            kernel_size=pooling_size,
            stride=1,
            padding=(round(pooling_size[0] / 2 + 0.1) - 1, round(pooling_size[1] / 2 + 0.1) - 1)
        )

        self.block4 = nn.Sequential(
            ResidualBlock(in_channels=2 * FT, out_channels=2 * FT, groups=FT)
        )


        self.after_conv = nn.Sequential(
            nn.LayerNorm([2 * FT, chans - 2, samples // 8]),
            nn.ELU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.after_conv_2 = nn.Sequential(
            nn.LayerNorm([2 * FT, chans - 2, samples // 8]),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(dropout_rate)
        )



        self.tcn_block = TemporalConvNet(num_inputs=64, num_channels=[2 * FT, 2 * FT],
                                         kernel_size=tcn_kernelSize, dropout=tcn_dropout)

        self.flatten = nn.Flatten()
        self.liner_cla = LinearWithConstraint(
            in_features=2 * FT,
            out_features=num_classes,
            max_norm=.25
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)

        x3 = self.block3(x)
        x3 = self.block3(x3)
        x3 = self.block3(x3)
        x3 = self.after_conv(x3)

        x4 = self.block4(x3)
        x = self.after_conv_2(x4)

        x = torch.squeeze(x, dim=2)
        x = self.tcn_block(x)
        x = x[:, :, -1]

        x = self.flatten(x)
        x = self.liner_cla(x)
        cls = self.softmax(x)

        return cls