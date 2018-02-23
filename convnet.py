import torch
import torchvision
import lib
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib import *
from networks import *
import torch.optim as optim

def accuracy(x, y, net):
    if len(x) != len(y):
        print("Something went wrong")
    correct = 0

    outputs = net(Variable(x))
    _, predicted = torch.max(outputs.data, 1)
        
    labels = Variable(y)
    correct += (predicted == labels.data).sum()
    return float(correct) / float(len(y))

def train(train_x, train_y, net, test_x, test_y):
    num_epochs = 10
    criterion = nn.CrossEntropyLoss()
    # change optimizer to be what you want
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    training_accuracy = []
    test_accuracy = []

    training_size = len(train_x)
    for epoch in range(num_epochs):
        print("Starting Epoch:", epoch, accuracy(test_x, test_y, net))
        for batch_number in range(0, 9):
            batch_x = train_x[int((batch_number * training_size)/10) : int((batch_number + 1) * training_size/10)]
            batch_y = train_y[int((batch_number * training_size)/10) : int((batch_number + 1) * training_size/10)]
            inputs, labels = Variable(batch_x), Variable(batch_y)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def main():
    easy_X, easy_Y = load_shape_data("easy_shape_dataset")
    easy_X = torch.from_numpy(easy_X).float()
    easy_Y = torch.from_numpy(easy_Y).long()
    print(len(easy_X), len(easy_Y))
    easy_net = ConvolutionalNet()

    hard_X, hard_Y = load_shape_data("hard_shape_dataset")
    hard_X = torch.from_numpy(hard_X).float()
    hard_Y = torch.from_numpy(hard_Y).long()
    print(len(hard_X), len(hard_Y))

    hard_net = ConvolutionalNet()

    all_X, all_Y = load_both_shape_data("easy_shape_dataset", "hard_shape_dataset")
    all_X = torch.from_numpy(all_X).float()
    all_Y = torch.from_numpy(all_Y).long()
    print(len(all_X), len(all_Y))

    print("Training Easy")
    train(easy_X, easy_Y, easy_net, all_X, all_Y)
    print("Training Hard")
    train(hard_X, hard_Y, hard_net, all_X, all_Y)
main()