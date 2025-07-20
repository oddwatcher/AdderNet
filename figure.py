import matplotlib.pyplot as plt
import re
from pathlib import Path
# 读取日志文件内容
log_file = Path(input("请输入日志文件路径："))
with open(log_file, 'r') as din:
    log_text = din.read()


train_pattern = r"Train - Epoch (\d+), Avg\. Loss: ([\d\.]+)"
test_pattern = r" Test Avg\. Loss: ([\d\.]+), Accuracy: ([\d\.]+)"

epochs = []
train_losses = []
test_losses = []
test_accuracies = []

# 解析日志
for line in log_text.strip().split('\n'):
    train_match = re.search(train_pattern, line)
    test_match = re.search(test_pattern, line)

    if train_match:
        epoch = int(train_match.group(1))
        loss = float(train_match.group(2))
        epochs.append(epoch)
        train_losses.append(loss)
    
    if test_match:
        loss = float(test_match.group(1))
        acc = float(test_match.group(2))
        test_losses.append(loss)
        test_accuracies.append(acc)

min_len = min(len(epochs), len(test_losses), len(test_accuracies))
epochs_plot = epochs[:min_len]
train_losses_plot = train_losses[:min_len]
test_losses_plot = test_losses[:min_len]
test_accuracies_plot = test_accuracies[:min_len]

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs[0, 0].plot(epochs_plot, train_losses_plot, label='Train Loss', marker='o')
axs[0, 0].plot(epochs_plot, test_losses_plot, label='Test Loss', marker='s')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].set_title('Loss vs Epoch')
axs[0, 0].legend()
axs[0, 0].grid(True)

axs[0, 1].plot(epochs_plot, test_accuracies_plot, label='Test Accuracy', color='g', marker='^')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Accuracy')
axs[0, 1].set_title('Accuracy vs Epoch')
axs[0, 1].legend()
axs[0, 1].grid(True)

ax1 = axs[1, 0]
ax2 = ax1.twinx()

# Train Loss（左侧y轴）
color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss', color=color)
ax1.plot(epochs_plot, train_losses_plot, label='Train Loss', marker='o', color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Test Accuracy（右侧y轴）
color = 'tab:green'
ax2.set_ylabel('Test Accuracy', color=color)
ax2.plot(epochs_plot, test_accuracies_plot, label='Test Accuracy', marker='s', color=color)
ax2.tick_params(axis='y', labelcolor=color)

# 设置标题、合并图例
ax1.set_title('Train Loss and Test Accuracy (Dual Y Axes)')
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

fig.tight_layout()
ax1.grid(True, linestyle='--', alpha=0.5)

# 第四个子图可以留空或注释掉
axs[1, 1].axis('off')  # 隐藏第四个空白图

# 显示图像 & 保存
plt.tight_layout()

plt.savefig(log_file.with_suffix(".png"), dpi=300)
plt.show()