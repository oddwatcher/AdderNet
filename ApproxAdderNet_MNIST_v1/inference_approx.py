import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from models.adder_trainable import Adder2D
import torch.nn as nn

# 推理用支持approx_bit的模型定义
class BasicBlockApprox(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, approx_bit=0):
        super().__init__()
        self.adder = Adder2D(in_channels, out_channels, kernel_size, stride, padding, approx_bit=approx_bit)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.adder(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class MNISTAdderNetApprox(nn.Module):
    def __init__(self, num_classes=10, approx_bit=0):
        super().__init__()
        self.layer1 = BasicBlockApprox(1, 16, 3, 1, 1, approx_bit=approx_bit)
        self.layer2 = BasicBlockApprox(16, 32, 3, 1, 1, approx_bit=approx_bit)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

def evaluate(model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1. 浮点权重推理
    print("==> 浮点权重精确推理:")
    model_fp = MNISTAdderNetApprox(approx_bit=0).to(device)
    state_dict = torch.load("mnist_addernet.pth", map_location=device)
    model_fp.load_state_dict(state_dict)
    acc_fp = evaluate(model_fp, device)
    print(f"浮点精确推理准确率: {acc_fp:.2f}%\n")

    # 2. 量化权重推理（支持approx_bit sweep）
    print("==> 量化权重+近似加法推理:")
    # 权重量化为int32
    quantized_state_dict = {}
    scale_dict = {}
    qmax = 2**31 - 1
    qmin = -2**31
    for k, v in state_dict.items():
        if torch.is_floating_point(v):
            max_abs = v.abs().max().item()
            scale = max_abs / qmax if max_abs != 0 else 1.0
            v_int = torch.clamp((v / scale).round(), qmin, qmax).to(torch.int32)
            quantized_state_dict[k] = v_int
            scale_dict[k] = scale
        else:
            quantized_state_dict[k] = v
    # sweep不同approx_bit
    bits = list(range(0, 33, 2))
    accs = []
    for bit in bits:
        print(f"approx_bit = {bit}")
        model_q = MNISTAdderNetApprox(approx_bit=bit).to(device)
        # 还原权重为浮点（int32*scale）
        restored_state_dict = {}
        for k, v in quantized_state_dict.items():
            if k in scale_dict:
                restored_state_dict[k] = (v.float() * scale_dict[k])
            else:
                restored_state_dict[k] = v
        model_q.load_state_dict(restored_state_dict)
        acc = evaluate(model_q, device)
        accs.append(acc)
        print(f"  准确率: {acc:.2f}%")
    # 绘图
    plt.plot(bits, accs, marker='o', label="Quantized+Approx")
    plt.axhline(acc_fp, color='r', linestyle='--', label="Float32 Baseline")
    plt.xlabel("Approximate Bits")
    plt.ylabel("Test Accuracy (%)")
    plt.title("MNIST Accuracy vs Approximate Bit Width")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mnist_inference_approx_sweep.png")
    plt.show()

if __name__ == "__main__":
    main()
