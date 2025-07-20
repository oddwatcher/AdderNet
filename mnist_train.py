# Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import os
from resnet20mono import resnet20
import torch
from torch.autograd import Variable
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import math
import time
from tqdm import tqdm


parser = argparse.ArgumentParser(description="train-addernet")

# Basic model parameters.
parser.add_argument("--data", type=str, default="./cache/data/")
parser.add_argument("--output_dir", type=str, default="./cache/models/")
args = parser.parse_args()
start_time = time.asctime()
log = args.output_dir + start_time + "log.txt"
os.makedirs(args.output_dir, exist_ok=True)

acc = 0
acc_best = 0
weights_best = {}

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ]
)

data_train = MNIST(args.data, download=True, transform=transform_train)
data_test = MNIST(args.data, train=False, transform=transform_test)

data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=16)
data_test_loader = DataLoader(data_test, batch_size=100, num_workers=0)

net = resnet20().cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)


def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    lr = 0.05 * (1 + math.cos(float(epoch) / 400 * math.pi))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train(epoch, start_time):
    adjust_learning_rate(optimizer, epoch)
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    print("Train - Epoch %d" % epoch)
    for i, (images, labels) in tqdm(
        enumerate(data_train_loader), total=len(data_train_loader)
    ):
        images, labels = Variable(images).cuda(), Variable(labels).cuda()

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
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
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
        torch.save(net.state_dict(), args.output_dir + "addernet_mono_%d.pt" % epoch)
        torch.save(weights_best, args.output_dir + "addernet_mono_best.pt")
    else:
        torch.save(net.state_dict(), args.output_dir + "addernet_mono_temp.pt")


def main():
    epoch = 400
    for e in range(1, epoch):
        train_and_test(e, time.time())
    torch.save(net.state_dict(), args.output_dir + "addernet_mono_final.pt")
    torch.save(weights_best, args.output_dir + "addernet_mono_best.pt")


if __name__ == "__main__":
    main()
