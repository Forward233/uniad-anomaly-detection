# UniAD 代码学习指南

本指南旨在帮助开发者深入理解 UniAD 异常检测框架的代码实现和核心原理。

## 📚 文档导航

### 基础部分
- **[部署文档](DEPLOYMENT_CN.md)** - 环境配置、数据集准备、训练部署流程

### 架构模块
- **[整体架构](architecture/ARCHITECTURE.md)** - 项目架构设计、三层结构、数据流程
- **[模型详解](architecture/MODELS.md)** - Backbone、Neck、Reconstruction模块深度解析
- **[数据处理](architecture/DATASETS.md)** - 数据集加载、预处理、数据增强

### 训练模块
- **[训练流程](training/TRAINING.md)** - 训练主流程、配置文件、损失函数、优化器
- **[评估可视化](training/EVALUATION.md)** - 评估指标、可视化工具、结果分析

### 进阶内容
- **[进阶实践](advanced/ADVANCED.md)** - 自定义数据集、修改网络、调试技巧、性能优化

## 🎯 学习路径

### 初学者路径
1. **环境搭建** → [部署文档](DEPLOYMENT_CN.md)
2. **理解架构** → [整体架构](architecture/ARCHITECTURE.md)
3. **运行训练** → [训练流程](training/TRAINING.md)
4. **查看结果** → [评估可视化](training/EVALUATION.md)

### 研究者路径
1. **架构设计** → [整体架构](architecture/ARCHITECTURE.md)
2. **模型原理** → [模型详解](architecture/MODELS.md)
3. **数据处理** → [数据处理](architecture/DATASETS.md)
4. **训练细节** → [训练流程](training/TRAINING.md)
5. **实验分析** → [评估可视化](training/EVALUATION.md)

### 开发者路径
1. **快速上手** → [部署文档](DEPLOYMENT_CN.md)
2. **代码架构** → [整体架构](architecture/ARCHITECTURE.md)
3. **核心模块** → [模型详解](architecture/MODELS.md)
4. **自定义开发** → [进阶实践](advanced/ADVANCED.md)

## 📖 项目概述

### 什么是 UniAD？

UniAD (Unified Anomaly Detection) 是一个统一的多类别异常检测框架，发表于 NeurIPS 2022。它能够：

- 使用单一模型处理多个类别的异常检测
- 无需为每个类别单独训练模型
- 支持工业缺陷检测（MVTec-AD）和图像异常检测（CIFAR-10）

### 核心思想

**特征重建 + Transformer**：
- 使用预训练的骨干网络提取特征
- 通过 Transformer 重建正常样本的特征
- 重建误差作为异常分数

**关键技术**：
- Feature Jitter：训练时对特征添加扰动，增强鲁棒性
- Neighbor Mask：限制自注意力的邻域范围，保持局部性
- 位置编码：为特征添加空间位置信息

### 代码结构

```
UniAD/
├── models/                    # 模型定义
│   ├── backbones/            # 骨干网络（ResNet、EfficientNet）
│   ├── necks/                # 特征融合（MFCN）
│   └── reconstructions/      # 重建网络（UniAD、Transformer）
├── datasets/                  # 数据集加载
│   ├── custom_dataset.py     # MVTec-AD数据集
│   ├── cifar_dataset.py      # CIFAR-10数据集
│   └── transforms.py         # 数据增强
├── tools/                     # 训练评估工具
│   └── train_val.py          # 训练主脚本
├── utils/                     # 工具函数
│   ├── eval_helper.py        # 评估工具
│   ├── vis_helper.py         # 可视化工具
│   └── misc_helper.py        # 辅助函数
└── experiments/               # 实验配置
    ├── MVTec-AD/             # MVTec-AD实验
    └── CIFAR-10/             # CIFAR-10实验
```

## 🔍 快速开始

### 1. 环境配置

```bash
# 创建环境
conda create -n uniad python=3.8
conda activate uniad

# 安装PyTorch
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# 安装依赖
pip install einops opencv-python Pillow PyYAML scikit-learn scipy tabulate tensorboardX easydict
pip install protobuf==3.20.3
```

详见 [部署文档](DEPLOYMENT_CN.md)

### 2. 运行训练

```bash
cd experiments/MVTec-AD
bash train_torch.sh 1 0
```

### 3. 查看日志

```bash
tail -f log/train.log
```

## 📊 数据集

### MVTec-AD
工业异常检测数据集，包含15个物体类别的正常和异常样本。

**下载**: https://www.mvtec.com/company/research/datasets/mvtec-ad

### CIFAR-10
图像分类数据集，改造用于异常检测（将部分类别作为正常，其他作为异常）。

**下载**: https://www.cs.toronto.edu/~kriz/cifar.html

## 🎓 核心概念

### 1. 三层架构

```
输入图像 → Backbone → Neck → Reconstruction → 重建特征 → 异常分数
```

- **Backbone**: 提取图像特征（ResNet、EfficientNet）
- **Neck**: 融合多尺度特征（MFCN）
- **Reconstruction**: 重建正常特征（UniAD Transformer）

### 2. Transformer 重建

```python
# 伪代码
features = backbone(image)              # 特征提取
features = neck(features)               # 特征融合
queries = learnable_queries             # 可学习的查询向量
reconstructed = transformer(queries, features)  # Transformer重建
loss = MSE(features, reconstructed)     # 重建损失
```

### 3. 异常检测流程

**训练阶段**：
1. 只使用正常样本训练
2. 学习重建正常样本的特征
3. 最小化重建误差

**测试阶段**：
1. 对测试样本进行特征重建
2. 计算重建误差
3. 高误差 = 异常，低误差 = 正常

## 🔧 常用命令

### 训练
```bash
# MVTec-AD
cd experiments/MVTec-AD
bash train_torch.sh 1 0

# CIFAR-10
cd experiments/CIFAR-10/01234
bash train_torch.sh 1 0
```

### 评估
```bash
cd experiments/MVTec-AD
bash eval_torch.sh 1 0
```

### 可视化
```bash
# 查看重建结果
python tools/vis_recon.py --config experiments/MVTec-AD/config.yaml

# 查看查询向量
python tools/vis_query.py --config experiments/MVTec-AD/config.yaml
```

## 📝 论文引用

```bibtex
@inproceedings{you2022unified,
  title={A Unified Model for Multi-class Anomaly Detection},
  author={You, Zhiyuan and Cui, Lei and Shen, Yujun and Yang, Kai and Lu, Xin and Zheng, Yu and Le, Xinyi},
  booktitle={NeurIPS},
  year={2022}
}
```

## 🔗 相关资源

- **原始项目**: https://github.com/zhiyuanyou/UniAD
- **本项目**: https://github.com/Forward233/uniad-anomaly-detection
- **论文**: https://arxiv.org/abs/2206.03687

## 💡 学习建议

1. **先运行再理解**：先成功运行训练，观察输出，再深入代码
2. **模块化学习**：按照架构→数据→训练→评估的顺序学习
3. **实验对比**：尝试不同参数配置，观察结果变化
4. **代码调试**：使用IDE断点调试，理解数据流向
5. **阅读论文**：结合论文理解代码实现细节

## ❓ 常见问题

### Q: 训练需要多长时间？
A: MVTec-AD约12-13小时（1000 epochs），但通常100-300 epochs即可达到较好效果。

### Q: 显存不足怎么办？
A: 减小batch_size或input_size，参考[部署文档](DEPLOYMENT_CN.md)第7.4节。

### Q: 如何添加自己的数据集？
A: 参考[进阶实践](advanced/ADVANCED.md)中的自定义数据集章节。

### Q: 如何理解Feature Jitter？
A: 参考[模型详解](architecture/MODELS.md)中的Feature Jitter章节。

---

**开始学习**：建议从 [整体架构](architecture/ARCHITECTURE.md) 开始，了解项目的整体设计思路。
