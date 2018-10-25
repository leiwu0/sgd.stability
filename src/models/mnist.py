import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(1,6,5,stride=1) # 28-5+1=24
        self.conv2 = nn.Conv2d(6,16,5,stride=1) # 12-5+1=8
        self.fc1 = nn.Linear(4*4*16,200)
        self.fc2 = nn.Linear(200,10)

    def forward(self,x):
        if x.ndimension()==3:
            x = x.unsqueeze(0)
        o = F.relu(self.conv1(x))
        o = F.avg_pool2d(o,2,2)

        o = F.relu(self.conv2(o))
        o = F.avg_pool2d(o,2,2)

        o = o.view(o.shape[0],-1)
        o = self.fc1(o)
        o = F.relu(o)
        o = self.fc2(o)
        return o

class FNN(nn.Module):
    def __init__(self):
        super(FNN,self).__init__()
        self.net = nn.Sequential(nn.Linear(784,500),
                            nn.ReLU(),
                            nn.Linear(500,500),
                            nn.ReLU(),
                            nn.Linear(500,500),
                            nn.ReLU(),
                            nn.Linear(500,10))

    def forward(self,x):
        x = x.view(x.shape[0],-1)
        o = self.net(x)
        return o


def lenet():
    return LeNet()

def fnn():
    return FNN()

