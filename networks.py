import torch.nn as nn
import torch.nn.functional as F

class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.fc1 = nn.Linear(785, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class FullyConnectedNet(nn.Module):
    def __init__(self):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(50*50, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 3)
        
    def forward(self, x):
        x = x.view(-1,50*50)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ShallowConvNet(nn.Module):
    def __init__(self):
        super(ShallowConvNet, self).__init__()
        self.convr1 = nn.Conv2d(1, 6, 5,padding=2) # in channels, out channels, kernel size.
        self.convr1_bn = nn.BatchNorm2d(6)
        self.conv1 = nn.Conv2d(6,6,5,padding=2)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2) # kernel size
        self.fc1 = nn.Linear(6 * 25 * 25, 3)

    def forward(self,x):
        x=self.conv1_bn(self.conv1(F.relu(self.convr1_bn(self.convr1(x)))))
        x=self.pool(x)
        x=x.view(-1, 6 * 25 * 25)
        x = F.relu(self.fc1(x))
        return x

class ConvolutionalNet(nn.Module):
    def __init__(self):
        super(ConvolutionalNet, self).__init__()
        self.convr1 = nn.Conv2d(1, 3, 5,padding=2) # in channels, out channels, kernel size.
        self.convr1_bn = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3,6,5,padding=2)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2) # kernel size
        self.convr2 = nn.Conv2d(6,6,5,padding=2)
        self.convr2_bn = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5,padding=2)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(2304, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self,x):
        x=self.conv1_bn(self.conv1(F.relu(self.convr1_bn(self.convr1(x)))))
        x=self.pool(x)
        x=self.conv2_bn(self.conv2(F.relu(self.convr2_bn(self.convr2(x)))))
        x=self.pool(x)
        x=x.view(-1,16*12*12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
