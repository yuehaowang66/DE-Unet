""" Full assembly of the parts to form the complete network """

from .block import *


class DEUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(DEUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = Up(1024, 512, 256, 128, 512, bilinear)
        self.up2 = Up(512, 256, 128, 64, 256, bilinear)
        self.up3 = Up(256, 128, 64, 64, 128, bilinear)
        self.up4 = Up(128, 64, 64, 64, 64, bilinear)

        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        input_size = x.shape[2:]   
        x1 = self.inc(x)      # [B,64,H,W]
        x2 = self.down1(x1)   # [B,128,H/2,W/2]
        x3 = self.down2(x2)   # [B,256,H/4,W/4]
        x4 = self.down3(x3)   # [B,512,H/8,W/8]
        x5 = self.down4(x4)   # [B,1024,H/16,W/16]

        x = self.up1(x5, [x4, x3, x2])
        x = self.up2(x, [x3, x2, x1])
        x = self.up3(x, [x2, x1, x1])
        x = self.up4(x, [x1, x1, x1])

        logits = self.outc(x)
        logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)

        return logits



    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)