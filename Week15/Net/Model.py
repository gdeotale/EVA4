from torch import nn
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=2, dilation=2, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Upsample(nn.Module):
    def __init__(self,  scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) #126
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        ) #124
        self.convblock2_ = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ) #122
        
        self.pool1 = nn.MaxPool2d(2) #128
        
        self.convblock3 = nn.Sequential(
            depthwise_separable_conv(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.convblock3_ = nn.Sequential(
            depthwise_separable_conv(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ) 
        self.convblock3_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), bias=False),
        ) 
        
        self.pool2 = nn.MaxPool2d(2) #64

        self.convblock4 = nn.Sequential(
            depthwise_separable_conv(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.convblock4_ = nn.Sequential(
            depthwise_separable_conv(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ) 
        self.convblock4_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), bias=False),
        ) 
        
        self.convblock5 = nn.Sequential(
            depthwise_separable_conv(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ) 
        self.convblock5_ = nn.Sequential(
            depthwise_separable_conv(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ) 
        self.convblock5_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), bias=False),
        )

        self.convblock6 = nn.Sequential(
            depthwise_separable_conv(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ) 
        self.convblock6_ = nn.Sequential(
            depthwise_separable_conv(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ) 
        self.convblock6_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), bias=False),
        )
        
        self.upscale1 = nn.Upsample(64)
        
        self.convblock7 = nn.Sequential(
            depthwise_separable_conv(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.convblock7_ = nn.Sequential(
            depthwise_separable_conv(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.convblock7_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), bias=False),
        )

        self.upscale2 = nn.Upsample(128)
        
        self.convblock8 = nn.Sequential(
            depthwise_separable_conv(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.convblock8_ = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(3, 3), padding=1, bias=False),
        ) 

    def forward(self, sample):
        f1 = sample["f1_image"]
        f2 = sample["f2_image"]

        f1 = self.pool1(self.convblock2_(self.convblock2(self.convblock1(f1))))
        f2 = self.pool1(self.convblock2_(self.convblock2(self.convblock1(f2))))
        f = torch.cat([f1, f2], dim=1)
        fx = (self.convblock4_(self.convblock4(f)))
        f = f+fx
        f = self.pool2(f)
        fx = self.convblock5_1x1(self.convblock5_(self.convblock5(f)))
        f = f + fx

        #Net1
        fg = self.upscale1(f)
        fx = self.convblock7_1x1(self.convblock7_(self.convblock7(fg)))
        f0 = fg + fx
        f0 = self.upscale2(f0)
        f0 = self.convblock8_(self.convblock8(f0))
        #Net2
        fx = self.convblock6_1x1(self.convblock6_(self.convblock6(f)))
        fa = f + fx
        fa = self.upscale1(fa)
        fx_ = self.convblock7_1x1(self.convblock7_(self.convblock7(fa)))
        f_ = fa + fx_
        f_ = self.upscale2(f_)
        f_ = self.convblock8_(self.convblock8(f_))
        return f0, f_