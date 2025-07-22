from resnet20 import resnet20
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.autograd import Variable

device = "cpu"
net = resnet20()
qint8 =torch.load("Resnet20_adder_CIFAR_int8_quantized.pt", weights_only=True)
net.load_state_dict(
    qint8
)
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
data_test_loader = DataLoader(
    CIFAR10("cache/data/", train=True, transform=transform_test),
    batch_size=100,
    num_workers=0,
    shuffle=True,
)
criterion = torch.nn.CrossEntropyLoss().to(device)


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

    avg_loss /= len(data_test_loader)
    acc = float(total_correct) / len(data_test_loader)
    if acc_best < acc:
        acc_best = acc
        weights_best = net.state_dict()
    print("Test Avg. Loss: %f, Accuracy: %f" % (avg_loss.data.item(), acc))

if __name__ == "__main__":
    test()