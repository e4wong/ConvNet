import torch
import torchvision
import lib
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib import *
from networks import *
import torch.optim as optim

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

def accuracy(x, y, net):
    if len(x) != len(y):
        print("Something went wrong")
    correct = 0

    outputs = net(Variable(x))
    _, predicted = torch.max(outputs.data, 1)
        
    labels = Variable(y)
    correct += (predicted == labels.data).sum()
    return float(correct) / float(len(y))

def train(x, y, net):
    num_epochs = 10
    criterion = nn.CrossEntropyLoss()
    # change optimizer to be what you want
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    training_accuracy = []
    test_accuracy = []

    training_size = len(x)
    for epoch in range(num_epochs):
        print("Starting Epoch:", epoch, accuracy(x, y, net))
        for batch_number in range(0, 9):
            batch_x = x[int((batch_number * training_size)/10) : int((batch_number + 1) * training_size/10)]
            batch_y = y[int((batch_number * training_size)/10) : int((batch_number + 1) * training_size/10)]
            inputs, labels = Variable(batch_x), Variable(batch_y)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def main():
    X,Y = load_shape_data("shape_dataset")
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).long()

    net = ConvolutionalNet()
    train(X, Y, net)

def main1():
    training_set_x, training_set_y = load_data("training_set")
    training_set_x = torch.from_numpy(training_set_x).float()
    training_set_y = torch.from_numpy(training_set_y).long()

    net = BasicNet()
    train(training_set_x, training_set_y, net)

main()