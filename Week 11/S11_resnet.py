import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.prep = self.convlayer(3, 64, 1) # output_size = 32x32x64
        #Layer1
        self.X =  self.convblock(64,128, 1)# output_size = 16x16x128
        self.resX = self.resblock(128,128)
        #Layer2
        self.X1 =  self.convblock(128,256, 1)# output_size = 16x16x128
        #Layer3
        self.X2 =  self.convblock(256,512, 1)# output_size = 16x16x128
        self.resX2 = self.resblock(512,512)
        self.maxlayer = nn.MaxPool2d(kernel_size=4)
        self.fc = nn.Linear(512, 10)

    def convlayer(self, inx, outx, padx):
      return nn.Sequential(
            nn.Conv2d(in_channels=inx, out_channels=outx, kernel_size=(3, 3), padding=padx, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def convblock(self, inx, outx, padx):
      return nn.Sequential(
            nn.Conv2d(in_channels=inx, out_channels=outx, kernel_size=(3, 3), padding =padx, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(outx),
            nn.ReLU(),
        )

    def resblock(self, inx, outx):
      return nn.Sequential(
            nn.Conv2d(in_channels=inx, out_channels=outx, kernel_size=(3, 3), padding =1, bias=False),
            nn.BatchNorm2d(outx),
            nn.ReLU(),
            nn.Conv2d(in_channels=inx, out_channels=outx, kernel_size=(3, 3), padding =1, bias=False),
            nn.BatchNorm2d(outx),
            nn.ReLU(),
        )
        

    def forward(self, x):
        x = self.prep(x) 
        x = self.X(x) 
        x1 = self.resX(x)
        x = x + x1
        x = self.X1(x)
        x = self.X2(x)
        x1 = self.resX2(x)
        x = x + x1
        x = self.maxlayer(x)
        x = x.view(-1, 512)
        x = F.softmax(x)
        return x