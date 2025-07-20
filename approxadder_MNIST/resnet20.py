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

# 2020.01.10-Replaced conv with adder
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import adder
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return adder.adder2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Conv2d(64 * block.expansion, num_classes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_classes)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn2(x)

        return x.view(x.size(0), -1)


def resnet20(**kwargs):
    return ResNet(BasicBlock, [3, 3, 3], **kwargs)


data_train = MNIST(args.data, download=True, transform=transform_train)
data_test = MNIST(args.data, train=False, transform=transform_test)

data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=16)
data_test_loader = DataLoader(data_test, batch_size=100, num_workers=0)

net = resnet20().to(device)
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
        torch.save(net.state_dict(), args.output_dir + "resnet_conv_mono_%d.pt" % epoch)
        torch.save(weights_best, args.output_dir + "resnet_conv_mono_best.pt")
    else:
        torch.save(net.state_dict(), args.output_dir + "resnet_conv_temp.pt")


def main():
    epoch = 100
    for e in range(1, epoch):
        train_and_test(e, time.time())
    torch.save(net.state_dict(), args.output_dir + "resnet_conv_mono_final.pt")
    torch.save(weights_best, args.output_dir + "resnet_conv_mono_best.pt")


if __name__ == "__main__":
    main()
