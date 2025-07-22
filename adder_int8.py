from resnet20 import resnet20
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

net = resnet20()
net.load_state_dict(torch.load("trained/addernet_CIFAR10_best.pt",weights_only=True))

net.eval()
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
net.qconfig = torch.quantization.get_default_qconfig("fbgemm")
model_prepared = torch.quantization.prepare(net, inplace=False)

data_test_loader = DataLoader(
    CIFAR10("cache/data/", train=True, transform=transform_test),
    batch_size=100,
    num_workers=0,
    shuffle=True
)
i =0
with torch.no_grad():
    for img, label in tqdm(data_test_loader,total=100):
        i+=1
        if i >=100:
            break
        model_prepared(img)
model_quantized = torch.quantization.convert(model_prepared, inplace=False)

torch.jit.save(torch.jit.script(model_quantized), "model_int8_quantized.pth")
