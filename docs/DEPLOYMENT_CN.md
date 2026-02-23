# UniAD 运行部署文档

本文档详细说明 UniAD 异常检测项目的环境配置、数据集准备和训练部署流程。

## 1. 环境要求

### 硬件要求
- **GPU**: NVIDIA GPU（建议 RTX 3080 Ti 或更高，至少12GB显存）
- **内存**: 建议16GB以上
- **存储**: 至少20GB可用空间（用于数据集和模型）

### 软件要求
- **操作系统**: Linux（Ubuntu 18.04/20.04）或 Docker容器
- **CUDA**: 11.1
- **Python**: 3.8
- **PyTorch**: 1.8.1+cu111

## 2. 依赖安装

### 2.1 创建Python环境

推荐使用Conda创建独立环境：

```bash
conda create -n uniad python=3.8
conda activate uniad
```

### 2.2 安装PyTorch

```bash
# 安装PyTorch 1.8.1 with CUDA 11.1
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

### 2.3 安装系统依赖

```bash
# Ubuntu/Debian系统
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### 2.4 安装Python依赖包

```bash
cd UniAD

# 安装核心依赖
pip install einops==0.4.1
pip install numpy==1.19.5
pip install opencv-python==3.4.10.37
pip install Pillow==8.3.2
pip install PyYAML==5.3.1
pip install scikit-learn==0.24.2
pip install scipy==1.5.4
pip install tabulate==0.8.10
pip install tensorboardX==1.8
pip install easydict

# 重要：降级protobuf以兼容tensorboardX
pip install protobuf==3.20.3
```

**注意**：
- 不要使用 `pip install -r requirements.txt` 一键安装，因为其中的PyTorch版本较旧
- 必须降级protobuf到3.20.3，否则tensorboardX会报错

### 2.5 验证环境

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import cv2, tensorboardX, einops, easydict"
```

应该看到CUDA可用且所有包导入无错误。

## 3. 数据集准备

### 3.1 MVTec-AD 数据集

#### 下载
从官方网站下载：https://www.mvtec.com/company/research/datasets/mvtec-ad

或使用命令：
```bash
cd data/MVTec-AD
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
```

#### 解压
```bash
cd data/MVTec-AD
tar -xf mvtec_anomaly_detection.tar.xz
```

#### 验证目录结构
```bash
data/MVTec-AD/
├── mvtec_anomaly_detection/
│   ├── bottle/
│   │   ├── train/
│   │   ├── test/
│   │   └── ground_truth/
│   ├── cable/
│   ├── capsule/
│   ├── carpet/
│   ├── grid/
│   ├── hazelnut/
│   ├── leather/
│   ├── metal_nut/
│   ├── pill/
│   ├── screw/
│   ├── tile/
│   ├── toothbrush/
│   ├── transistor/
│   ├── wood/
│   └── zipper/
├── train.json
└── test.json
```

### 3.2 CIFAR-10 数据集

#### 下载
```bash
cd data/CIFAR-10
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

#### 解压
```bash
tar -xzf cifar-10-python.tar.gz
```

#### 验证目录结构
```bash
data/CIFAR-10/
└── cifar-10-batches-py/
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    ├── test_batch
    ├── batches.meta
    └── readme.html
```

## 4. 配置说明

### 4.1 MVTec-AD 配置

配置文件位置：`experiments/MVTec-AD/config.yaml`

关键参数：
```yaml
dataset:
  batch_size: 8              # 批次大小，根据显存调整
  workers: 4                 # 数据加载线程数
  
trainer:
  max_epoch: 1000           # 最大训练轮数
  val_freq_epoch: 10        # 验证频率
  optimizer:
    type: AdamW
    kwargs:
      lr: 0.0001            # 学习率
      betas: [0.9, 0.999]   # Adam参数
      weight_decay: 0.0001
```

### 4.2 CIFAR-10 配置

CIFAR-10提供4种实验配置，分别使用不同类别作为正常样本：

- **01234**: 类别 0,1,2,3,4 为正常（airplane, automobile, bird, cat, deer）
- **02468**: 类别 0,2,4,6,8 为正常（airplane, bird, deer, frog, ship）
- **13579**: 类别 1,3,5,7,9 为正常（automobile, cat, dog, horse, truck）
- **56789**: 类别 5,6,7,8,9 为正常（dog, frog, horse, ship, truck）

配置文件位置：`experiments/CIFAR-10/{01234,02468,13579,56789}/config.yaml`

**重要修复**：
CIFAR-10原始配置文件缺少AdamW的betas参数，需要添加：
```yaml
optimizer:
  type: AdamW
  kwargs:
    lr: 0.0001
    betas: [0.9, 0.999]      # 必须添加此行
    weight_decay: 0.0001
```

### 4.3 显存优化

如果显存不足，可以调整以下参数：
- 减小 `batch_size`（如从8降到4）
- 减小 `input_size`（如从[224,224]降到[192,192]）

## 5. 训练命令

### 5.1 MVTec-AD 训练

```bash
cd experiments/MVTec-AD

# 使用1个GPU训练
bash train_torch.sh 1 0

# 使用多个GPU训练（例如2个GPU）
bash train_torch.sh 2 0,1
```

参数说明：
- 第1个参数：GPU数量
- 第2个参数：CUDA设备ID（单GPU为0，多GPU为0,1,2等）

### 5.2 CIFAR-10 训练

```bash
cd experiments/CIFAR-10/01234  # 或其他配置目录

# 使用1个GPU训练
bash train_torch.sh 1 0
```

### 5.3 训练日志

训练过程中会显示：
```
Epoch: [1/1000]  Iter: [1/454000]  Time 0.10 (0.10)  Data 0.00 (0.00)  Loss 32.88 (32.88)  LR 0.00010
```

- **Epoch**: 当前轮数/总轮数
- **Iter**: 当前迭代/总迭代数
- **Time**: 每次迭代耗时
- **Data**: 数据加载耗时
- **Loss**: 当前损失值
- **LR**: 当前学习率

### 5.4 保存日志到文件

```bash
bash train_torch.sh 1 0 2>&1 | tee train.log
```

### 5.5 后台运行

```bash
nohup bash train_torch.sh 1 0 > train.log 2>&1 &
```

查看训练进度：
```bash
tail -f train.log
```

## 6. 评估和推理

### 6.1 模型评估

训练过程中每10个epoch自动进行验证，查看结果：
```bash
cat log/train.log | grep "mean_pixel_auc"
```

### 6.2 使用训练好的模型

评估脚本：
```bash
cd experiments/MVTec-AD
bash eval_torch.sh 1 0
```

### 6.3 结果文件

训练和评估结果保存在：
- **Checkpoints**: `experiments/*/checkpoints/`
- **日志**: `experiments/*/log/`
- **评估结果**: `experiments/*/result_eval_temp/`
- **可视化**: `experiments/*/vis_compound/`

## 7. 常见问题

### 7.1 AdamW优化器报错

**问题**：
```
UnboundLocalError: local variable 'beta1' referenced before assignment
```

**原因**：CIFAR-10配置文件缺少betas参数

**解决**：
在 `experiments/CIFAR-10/*/config.yaml` 的optimizer.kwargs中添加：
```yaml
betas: [0.9, 0.999]
```

### 7.2 OpenCV导入错误

**问题**：
```
ImportError: libGL.so.1: cannot open shared object file
```

**解决**：
```bash
apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### 7.3 protobuf版本冲突

**问题**：
```
AttributeError: module 'google.protobuf.descriptor' has no attribute '_internal_create_key'
```

**解决**：
```bash
pip install protobuf==3.20.3
```

### 7.4 显存不足

**问题**：
```
RuntimeError: CUDA out of memory
```

**解决**：
1. 减小batch_size（如从8改为4）
2. 减小input_size（如从[224,224]改为[192,192]）
3. 使用梯度累积
4. 使用显存更大的GPU

### 7.5 数据集路径错误

**问题**：
```
FileNotFoundError: [Errno 2] No such file or directory: '../../data/MVTec-AD/train.json'
```

**解决**：
确保当前目录在 `experiments/MVTec-AD/` 或 `experiments/CIFAR-10/*/`，并检查数据集是否正确解压。

### 7.6 训练速度慢

**优化建议**：
1. 增加 `workers` 参数（数据加载线程数）
2. 使用SSD存储数据集
3. 检查GPU利用率（使用 `nvidia-smi` 命令）

## 8. 目录结构说明

```
UniAD/
├── data/                          # 数据集目录
│   ├── MVTec-AD/                  # MVTec-AD数据集
│   │   ├── mvtec_anomaly_detection/  # 图像数据
│   │   ├── train.json             # 训练元数据
│   │   └── test.json              # 测试元数据
│   └── CIFAR-10/                  # CIFAR-10数据集
│       └── cifar-10-batches-py/   # 图像数据
├── datasets/                      # 数据集加载代码
│   ├── custom_dataset.py          # MVTec-AD数据集
│   └── cifar_dataset.py           # CIFAR-10数据集
├── experiments/                   # 实验配置和脚本
│   ├── MVTec-AD/
│   │   ├── config.yaml            # 配置文件
│   │   ├── train_torch.sh         # 训练脚本
│   │   ├── eval_torch.sh          # 评估脚本
│   │   ├── checkpoints/           # 模型checkpoint（训练时生成）
│   │   ├── log/                   # 训练日志（训练时生成）
│   │   └── result_eval_temp/      # 评估结果（评估时生成）
│   └── CIFAR-10/
│       ├── 01234/                 # 实验配置1
│       ├── 02468/                 # 实验配置2
│       ├── 13579/                 # 实验配置3
│       └── 56789/                 # 实验配置4
├── models/                        # 模型定义
│   ├── backbones/                 # 骨干网络
│   ├── necks/                     # 特征融合网络
│   └── reconstructions/           # 重建网络
├── tools/                         # 训练和评估工具
│   └── train_val.py               # 主训练脚本
├── utils/                         # 工具函数
└── requirements.txt               # 依赖列表

```

## 9. 性能参考

### 训练时间估算（RTX 3080 Ti 12GB）

| 数据集 | Batch Size | 单个Epoch时间 | 1000 Epochs总时间 |
|--------|-----------|--------------|------------------|
| MVTec-AD | 8 | ~45秒 | ~12-13小时 |
| CIFAR-10 | 16 | ~3分钟 | ~50-56小时 |

**建议**：
- MVTec-AD：训练100-300个epoch通常足够
- CIFAR-10：训练100-300个epoch通常足够
- 观察验证指标，当mean_pixel_auc不再提升时可提前停止

### 验证指标

关键指标：
- **pixel_auc**: 像素级异常检测AUC
- **mean_auc**: 平均异常检测AUC
- **std_auc**: 标准异常检测AUC
- **max_auc**: 最大池化异常检测AUC

## 10. 进阶使用

### 10.1 调整学习率

编辑 `config.yaml`：
```yaml
trainer:
  optimizer:
    kwargs:
      lr: 0.0001  # 调整此值
  lr_scheduler:
    type: StepLR
    kwargs:
      step_size: 800  # 学习率衰减步长
      gamma: 0.1      # 学习率衰减系数
```

### 10.2 更换骨干网络

编辑 `config.yaml` 的net部分：
```yaml
net:
  - name: backbone
    type: models.backbones.resnet50  # 或 efficientnet_b4
    kwargs:
      pretrained: True
      outlayers: [1,2,3]  # 特征层选择
```

### 10.3 分布式训练

使用多个GPU加速训练：
```bash
# 使用4个GPU
bash train_torch.sh 4 0,1,2,3
```

## 11. 引用

如果使用本项目，请引用原始论文：

```bibtex
@inproceedings{you2022unified,
  title={A Unified Model for Multi-class Anomaly Detection},
  author={You, Zhiyuan and Cui, Lei and Shen, Yujun and Yang, Kai and Lu, Xin and Zheng, Yu and Le, Xinyi},
  booktitle={NeurIPS},
  year={2022}
}
```

## 12. 技术支持

- 原始项目：https://github.com/zhiyuanyou/UniAD
- 本项目：https://github.com/Forward233/uniad-anomaly-detection
- Issues：如遇到问题，请在GitHub提交Issue

---

**最后更新**：2026年2月
