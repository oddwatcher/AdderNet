
import os
from resnet20 import resnet20
import torch
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
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
parser.add_argument("--preweight", type=str, default=None)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

if not torch.cuda.is_available():
    device = "cpu"
if args.preweight:
    net = resnet20()
    net.load_state_dict(torch.load(args.preweight,weights_only=True,map_location=device))
    net.to(device)
else:
    net = resnet20().to(device)
start_time = time.asctime()
log = args.output_dir + start_time + "log.txt"
os.makedirs(args.output_dir, exist_ok=True)

acc = 0
acc_best = 0
weights_best = {}
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

data_train = CIFAR10(args.data, download=True, transform=transform_train)
data_test = CIFAR10(args.data, train=False, transform=transform_test)

data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=16)
data_test_loader = DataLoader(data_test, batch_size=100, num_workers=0)

criterion = torch.nn.CrossEntropyLoss().to(device)
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
        torch.save(weights_best, args.output_dir + "addernet_best.pt")
    else:
        torch.save(net.state_dict(), args.output_dir + "addernet_temp.pt")


def main():
    epoch = 400
    for e in range(1, epoch):
        train_and_test(e, time.time())
    torch.save(net.state_dict(), args.output_dir + "addernet_final.pt")
    torch.save(weights_best, args.output_dir + "addernet_best.pt")


if __name__ == "__main__":
    main()
