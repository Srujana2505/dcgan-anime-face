import torch
import torchvision
import torch.nn as nn
from torchvision.utils import save_image

#VARIABLES
batch_size = 1
z = 100
ngf = 64
ndf = 64
nc = 3
ngpu=1
device = torch.device("cuda:0" if torch.cuda.is_available() and ngpu>0 else "cpu")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
        
            #input is z
            nn.ConvTranspose2d(z, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
        
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
        
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
        
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
    )

    def forward(self, input):
        return self.main(input)

netG = Generator().to(device)
netG.load_state_dict(torch.load('pretrained/G.pth'))