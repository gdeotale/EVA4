import torch.nn as nn
import torch.nn.functional as F

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) # output_size = 32
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) # output_size = 32
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) # output_size = 32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=2, dilation = 2,  bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) # output_size = 32
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=2, dilation=2,  bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) # output_size = 32
        self.convblock5_ = nn.Sequential(
            depthwise_separable_conv(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) # output_size = 32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convblock6 = nn.Sequential(
            depthwise_separable_conv(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
        ) # output_size = 32
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=2, dilation=2,  bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
        ) # output_size = 32
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # output_size = 1
        self.fc1 = nn.Linear(128, 10)
        #self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.convblock1(x) #3
        x = self.convblock2(x) #5
        x = self.convblock3(x) #9
        x = self.pool(x) #10
        x = self.convblock4(x) #18
        x = self.convblock5(x) #26
        x = self.convblock5_(x) #26
        x = self.pool1(x) #28
        x = self.convblock6(x) #52
        x = self.convblock7(x) #44
        x = self.pool2(x)
        x = self.gap(x)
        x = x.view(-1, 128)
        x = (self.fc1(x))
        #x = (self.fc2(x))
        return x