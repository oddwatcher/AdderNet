# inference.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.mnist_addernet_train import MNISTAdderNet
import matplotlib.pyplot as plt

device = torch.device('cpu')

# 量化并保存权重为int32
def quantize_and_save_state_dict(state_dict, save_path, num_bits=32):
    import numpy as np
    quantized_state_dict = {}
    scale_dict = {}
    qmax = 2**(num_bits-1) - 1
    qmin = -2**(num_bits-1)
    for k, v in state_dict.items():
        if torch.is_floating_point(v):
            max_abs = v.abs().max().item()
            scale = max_abs / qmax if max_abs != 0 else 1.0
            v_int = torch.clamp((v / scale).round(), qmin, qmax).to(torch.int32)
            quantized_state_dict[k] = v_int
            scale_dict[k] = scale
        else:
            quantized_state_dict[k] = v
    # 保存量化权重和scale
    torch.save({'int32_state_dict': quantized_state_dict, 'scale_dict': scale_dict}, save_path)
    print(f"量化权重已保存到: {save_path}")

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 准确率评估函数
def evaluate(model):
    import time
    model.eval()
    correct = 0
    total = 0
    times = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            start = time.time()
            outputs = model(images)
            end = time.time()
            times.append(end - start)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            print(f"Batch processed: {labels.size(0)} images, Time taken: {times[-1]*1000:.2f} ms")
            print(f"当前准确率: {100. * correct / total:.2f}%")
    avg_time = sum(times) / len(times) if times else 0
    print(f"平均每张图片推理时间: {avg_time*1000:.2f} ms")
    print(f"本轮推理共处理图片数: {total}")
    return 100. * correct / total

# 记录近似位宽与准确率
bits = list(range(0,1, 1))
accs = []

# 加载训练好的精确模型权重
base_state_dict = torch.load("mnist_addernet_epoch600.pth", map_location=device)

# 量化并保存为int32权重文件
quantize_and_save_state_dict(base_state_dict, "mnist_addernet_epoch600_int32.pth", num_bits=32)

for bit in bits:
    print(f"Testing with approx_bit = {bit}")
    model = MNISTAdderNet(approx_bit=bit).to(device)
    model.load_state_dict(base_state_dict)
    acc = evaluate(model)
    accs.append(acc)
    print(f"Accuracy: {acc:.2f}%")

# 绘图
plt.plot(bits, accs, marker='o', label="Accuracy")
plt.xlabel("Approximate Bits")
plt.ylabel("Test Accuracy (%)")
plt.title("MNIST Accuracy vs Approximate Bit Width")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("mnist_inference_sweep.png")
plt.show()
