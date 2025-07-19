# train.py
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.mnist_addernet_train import MNISTAdderNet
import matplotlib.pyplot as plt
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# 拆分训练集和验证集
full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# 初始化模型

model = MNISTAdderNet().to(device)  # 训练时始终使用精确加法
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)


# 记录 loss 和 acc
loss_list = []
acc_list = []
val_acc_list = []


# 实时画图设置
plt.ion()
# fig, ax = plt.subplots(1, 2, figsize=(10, 4))

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

for epoch in range(1, 11):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)}: Loss = {loss.item():.4f}, Acc = {100.*correct/total:.2f}%")

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    loss_list.append(avg_loss)
    acc_list.append(accuracy)

    # 验证集评估
    val_acc = evaluate(model, val_loader)
    val_acc_list.append(val_acc)
    print(f"Epoch {epoch} 训练Loss = {avg_loss:.4f}, 训练Acc = {accuracy:.2f}%, 验证Acc = {val_acc:.2f}%")

    # 实时画图
    # ax[0].cla()
    # ax[0].plot(loss_list, label='Train Loss')
    # ax[0].set_xlabel("Epoch")
    # ax[0].set_ylabel("Loss")
    # ax[0].set_title("Training Loss")
    # ax[0].grid(True)
    # ax[0].legend()

    # ax[1].cla()
    # ax[1].plot(acc_list, label="Train Acc (%)", color="green")
    # ax[1].plot(val_acc_list, label="Val Acc (%)", color="orange")
    # ax[1].set_xlabel("Epoch")
    # ax[1].set_ylabel("Accuracy (%)")
    # ax[1].set_title("Accuracy Curve")
    # ax[1].grid(True)
    # ax[1].legend()

    # plt.tight_layout()
    # plt.pause(0.1)

# 保存模型
torch.save(model.state_dict(), "mnist_addernet.pth")
print("模型已保存为 mnist_addernet.pth")

# 保存最终曲线
# plt.ioff()
# fig.savefig("training_result.png")
# print("训练曲线已保存为 training_result.png")
# plt.show()