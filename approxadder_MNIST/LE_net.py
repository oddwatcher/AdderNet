import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import argparse
import os
import time
from tqdm import tqdm

parser = argparse.ArgumentParser(description="train-lenet")
# Basic model parameters.
parser.add_argument("--data", type=str, default="./cache/data/")
parser.add_argument("--output_dir", type=str, default="./cache/models/")
args = parser.parse_args()
start_time = time.asctime()
log = args.output_dir + start_time + "log.txt"
os.makedirs(args.output_dir, exist_ok=True)
device = "cuda:1"

acc = 0
acc_best = 0
weights_best = {}

transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
transform_test = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


def lenet():
    return LeNet()


data_train = MNIST(args.data, download=True, transform=transform_train)
data_test = MNIST(args.data, train=False, transform=transform_test)

data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=16)
data_test_loader = DataLoader(data_test, batch_size=100, num_workers=0)

net = lenet().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


def train(epoch, start_time):

    net.train()
    loss_list, batch_list = [], []
    print("Train - Epoch %d" % epoch)
    for i, (images, labels) in tqdm(
        enumerate(data_train_loader), total=len(data_train_loader)
    ):
        images, labels = Variable(images).to(device), Variable(labels).to(device)

        optimizer.zero_grad()

        output = net(images)

        loss = criterion(output, labels)

        loss_list.append(loss.data.item())
        batch_list.append(i + 1)
        loss.backward()
        optimizer.step()
    total_loss = 0
    for i in loss_list:
        total_loss += i
    avg_loss = total_loss / len(loss_list)
    with open(log, "a") as f:
        f.write(
            "Train - Epoch %d, Avg. Loss: %f, Time:%dmin:%dsec \n "
            % (
                epoch,
                avg_loss,
                int((time.time() - start_time) / 60),
                int((time.time() - start_time) % 60),
            ),
        )


def test():
    global acc, acc_best, weights_best, net
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in tqdm(
            enumerate(data_test_loader), total=(len(data_test_loader))
        ):
            images, labels = Variable(images).to(device), Variable(labels).to(device)
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)
    if acc_best < acc:
        acc_best = acc
        weights_best = net.state_dict()
    with open(log, "a") as f:
        f.write("Test Avg. Loss: %f, Accuracy: %f\n" % (avg_loss.data.item(), acc))
    print("Test Avg. Loss: %f, Accuracy: %f" % (avg_loss.data.item(), acc))


def train_and_test(epoch, start):
    train(epoch, start)
    test()
    if epoch % 10 == 0:
        torch.save(net.state_dict(), args.output_dir + "lenet_mono_%d.pt" % epoch)
        torch.save(weights_best, args.output_dir + "lenet_mono_best.pt")
    else:
        torch.save(net.state_dict(), args.output_dir + "lenet_mono_temp.pt")


def main():
    epoch = 100
    for e in range(1, epoch):
        train_and_test(e, time.time())
    torch.save(net.state_dict(), args.output_dir + "lenet_mono_final.pt")
    torch.save(weights_best, args.output_dir + "lenet_mono_best.pt")


if __name__ == "__main__":
    main()
