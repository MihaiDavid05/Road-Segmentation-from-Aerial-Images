from utils.network_modules import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, dropout=False, cut_last_convblock=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout = dropout
        self.cut_last_convblock = cut_last_convblock

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, self.dropout, first=True)
        self.down2 = Down(128, 256, self.dropout)
        # self.down3 = Down(256, 512, self.dropout)
        if not cut_last_convblock:
            self.down3 = Down(256, 512, self.dropout)
            self.down4 = Down(512, 1024 // factor, self.dropout)
            self.up1 = Up(1024, 512 // factor, bilinear)

        else:
            self.down3 = Down(256, 512 // factor, self.dropout)

        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        if not self.cut_last_convblock:
            x5 = self.down4(x4)
            x = self.up1(x5, x4, self.dropout)
        else:
            x = x4
        x = self.up2(x, x3, self.dropout)
        x = self.up3(x, x2, self.dropout)
        x = self.up4(x, x1, self.dropout)
        logits = self.outc(x)
        return logits
