import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Double Conv Block
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# Attention Block
# -------------------------
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# -------------------------
# U-Net with Attention
# -------------------------
class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(AttentionUNet, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder
        self.conv1 = DoubleConv(in_channels, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        self.conv4 = DoubleConv(256, 512)
        self.conv5 = DoubleConv(512, 1024)

        self.pool = nn.MaxPool2d(2)

        # Decoder with Attention
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(512, 512, 256)
        self.up_conv4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(256, 256, 128)
        self.up_conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(128, 128, 64)
        self.up_conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(64, 64, 32)
        self.up_conv1 = DoubleConv(128, 64)

        # Output layer
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x5 = self.conv5(self.pool(x4))

        # Decoder with attention
        d4 = self.up4(x5)
        x4 = self.att4(d4, x4)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        x3 = self.att3(d3, x3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        x2 = self.att2(d2, x2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.up1(d2)
        x1 = self.att1(d1, x1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.up_conv1(d1)

        return self.out_conv(d1)
