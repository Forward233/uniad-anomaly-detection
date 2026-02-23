# UniAD 异常检测实验

基于 UniAD (NeurIPS 2022) 的工业和图像异常检测模型训练实验。

## 项目概述

UniAD 是一个统一的异常检测框架，支持多种数据集和场景的异常检测任务。

## 环境配置

- **GPU**: NVIDIA RTX 3080 Ti (12GB)
- **CUDA**: 11.1
- **PyTorch**: 1.8.1+cu111
- **Python**: 3.8
- **主要依赖**:
  - einops==0.4.1
  - opencv-python==3.4.10.37
  - Pillow==8.3.2
  - PyYAML==5.3.1
  - scikit-learn==0.24.2
  - scipy==1.5.4
  - tensorboardX==1.8
  - protobuf==3.20.3

## 数据集

### MVTec-AD
工业异常检测数据集，包含15个物体类别：
- bottle, cable, capsule, carpet, grid
- hazelnut, leather, metal_nut, pill, screw
- tile, toothbrush, transistor, wood, zipper

### CIFAR-10
经典图像分类数据集，改造用于异常检测任务。提供4种实验配置：
- **01234**: 类别 0,1,2,3,4 为正常样本
- **02468**: 类别 0,2,4,6,8 为正常样本
- **13579**: 类别 1,3,5,7,9 为正常样本
- **56789**: 类别 5,6,7,8,9 为正常样本

## 主要修改

1. **修复 CIFAR-10 配置文件** (`experiments/CIFAR-10/*/config.yaml`)
   - 添加 AdamW 优化器缺失的 `betas: [0.9, 0.999]` 参数
   - 解决 PyTorch 1.8.1 的 `UnboundLocalError` 问题

2. **依赖包兼容性**
   - 降级 protobuf 到 3.20.3，解决 tensorboardX 兼容性问题
   - 安装系统依赖 libgl1-mesa-glx 支持 opencv-python

## 训练命令

### MVTec-AD
```bash
cd experiments/MVTec-AD
bash train_torch.sh 1 0
```

### CIFAR-10
```bash
cd experiments/CIFAR-10/01234  # 或其他配置
bash train_torch.sh 1 0
```

参数说明：
- 第1个参数：GPU数量
- 第2个参数：CUDA设备ID

## 训练参数

- **Batch Size**: MVTec-AD=8, CIFAR-10=16
- **Max Epoch**: 1000
- **Learning Rate**: 0.0001
- **Optimizer**: AdamW
- **Backbone**: EfficientNet-B4 (预训练)
- **验证频率**: 每10个epoch

## 评估指标

- **Pixel AUC**: 像素级异常检测准确率
- **Mean AUC**: 平均异常检测准确率
- **Max AUC**: 最大池化异常检测准确率

## 目录结构

```
UniAD/
├── data/
│   ├── MVTec-AD/
│   │   ├── mvtec_anomaly_detection/  # 数据集图片
│   │   ├── train.json                # 训练元数据
│   │   └── test.json                 # 测试元数据
│   └── CIFAR-10/
│       └── cifar-10-batches-py/      # CIFAR-10数据
├── experiments/
│   ├── MVTec-AD/
│   │   ├── config.yaml               # 配置文件
│   │   └── train_torch.sh            # 训练脚本
│   └── CIFAR-10/
│       ├── 01234/
│       ├── 02468/
│       ├── 13579/
│       └── 56789/
├── models/                            # 模型定义
├── datasets/                          # 数据集加载
└── tools/                             # 训练和评估工具

```

## 参考文献

```bibtex
@inproceedings{you2022unified,
  title={A Unified Model for Multi-class Anomaly Detection},
  author={You, Zhiyuan and Cui, Lei and Shen, Yujun and Yang, Kai and Lu, Xin and Zheng, Yu and Le, Xinyi},
  booktitle={NeurIPS},
  year={2022}
}
```

## 原始仓库

https://github.com/zhiyuanyou/UniAD

## License

本项目基于原 UniAD 项目进行实验和修改。
