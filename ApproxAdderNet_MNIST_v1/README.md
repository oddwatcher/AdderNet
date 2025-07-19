



# ApproxAdderNet-MNIST

本项目实现了基于多数门逻辑的近似加法神经网络（AdderNet）在 MNIST 数据集上的训练与推理，支持逐位近似加法器的灵活配置与精度分析，适用于低功耗/存内计算等场景的算法研究。

## 主要功能

- 支持 AdderNet 精确与近似加法推理，近似位宽可灵活 sweep
- 训练与推理流程分离，推理时可加载不同 epoch 的模型权重
- 量化/反量化流程严格对称，支持有符号补码加法
- 支持自定义近似加法逻辑或查表（truth table）实现
- 训练/推理过程均可保存 loss、accuracy 曲线与 sweep 结果

## 网络结构

模型结构如下：

```
Input: 1×28×28
↓
Adder2D(1→16, 3×3, stride=1, padding=1) → BatchNorm2d(16) → ReLU
↓
Adder2D(16→32, 3×3, stride=1, padding=1) → BatchNorm2d(32) → ReLU
↓
AdaptiveAvgPool2d(1,1)
↓
Linear(32→10)
↓
CrossEntropy Loss
```

每个 Adder2D 层可配置精确/近似加法，近似位宽由 approx_bit 控制。

## 目录结构

```
ApproxAdderNet_MNIST_v1_complete_fixed/
├── data/                  # MNIST数据集（自动下载，无需手动准备）
├── models/                # 网络结构与加法核心模块
│   ├── adder_trainable.py      # 训练用Adder2D模块（float精度）
│   ├── adder_approx.py         # 近似加法推理核心（量化/近似/反量化）
│   ├── mnist_addernet_train.py # MNIST网络结构定义
│   └── __init__.py
├── utils/                 # 工具函数与查表支持
│   ├── truth_table_utils.py    # 近似加法查表工具（可自定义）
│   └── __init__.py
├── train.py               # 训练脚本，保存loss/acc曲线和模型
├── inference.py           # 推理与近似位宽sweep脚本
├── requirements.txt       # 依赖包
├── mnist_addernet.pth     # 训练好的模型权重
├── mnist_addernet_epoch600.pth # 训练中间权重
├── training_result.png    # 训练曲线
├── mnist_inference_sweep.png   # sweep结果曲线
├── truth_table.json       # 近似加法查表（如有）
└── README.md
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
python train.py
```
训练过程中会自动保存 loss/accuracy 曲线和模型权重。

### 推理与 sweep 精度分析

```bash
python inference.py
```
支持不同近似位宽（approx_bit）下的推理准确率 sweep，自动绘制“近似位宽-准确率”曲线。

## 近似加法自定义

修改 `models/adder_approx.py` 中 `approx_full_adder_logic` 或 `approx_full_adder_table` 可自定义近似规则，支持查表（truth table）和逻辑两种模式。

## 参考文献

- Y. Chen et al., “AdderNet: Do We Really Need Multiplications in Deep Learning?”, CVPR 2020. [论文链接](https://arxiv.org/abs/1912.13200)

---
```

## 依赖安装

```bash
pip install -r requirements.txt
```

## 训练模型

```bash
python train.py
```
- 支持断点续训，每100 epoch自动保存权重
- 训练过程自动保存 loss/accuracy 曲线

## 推理与 sweep 精度分析

```bash
python inference.py
```
- 支持不同近似位宽（approx_bit）下的推理准确率 sweep
- 自动绘制“近似位宽-准确率”曲线

## 近似加法自定义

- 修改 `models/adder_approx.py` 中 `approx_full_adder_logic` 或 `approx_full_adder_table` 可自定义近似规则
- 支持查表（truth table）和逻辑两种模式

## 参考文献

- Y. Chen et al., “AdderNet: Do We Really Need Multiplications in Deep Learning?”, CVPR 2020. [论文链接](https://arxiv.org/abs/1912.13200)

---

如需添加新数据集、支持更多近似规则或硬件能耗建模，欢迎联系作者协作！

---

近似度参数appro：
appro=-1  浮点推理 

量化到int32后：appro=0-32  int32推理 

