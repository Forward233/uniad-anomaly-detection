# UniAD 数据处理

本文档详细介绍 UniAD 的数据集加载、预处理和数据增强流程。

## 1. 数据集概览

### 1.1 支持的数据集

| 数据集 | 类型 | 样本数 | 用途 |
|--------|------|--------|------|
| MVTec-AD | 工业缺陷检测 | ~5000 | 异常检测 |
| CIFAR-10 | 图像分类 | 60000 | 异常检测 |

### 1.2 目录结构

**MVTec-AD**：
```
data/MVTec-AD/
├── mvtec_anomaly_detection/
│   ├── bottle/
│   │   ├── train/good/          # 正常样本
│   │   ├── test/good/           # 正常测试样本
│   │   ├── test/broken_large/   # 异常样本
│   │   └── ground_truth/        # 像素级标注
│   ├── cable/
│   └── ...
├── train.json                   # 训练元数据
└── test.json                    # 测试元数据
```

**CIFAR-10**：
```
data/CIFAR-10/
└── cifar-10-batches-py/
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    ├── test_batch
    └── batches.meta
```

## 2. MVTec-AD 数据集

### 2.1 CustomDataset 实现

**文件位置**：`datasets/custom_dataset.py`

**核心代码**：
```python
class CustomDataset(BaseDataset):
    def __init__(self, image_reader, meta_file, training=True, ...):
        super().__init__()
        self.image_reader = image_reader
        self.training = training
        
        # 加载元数据JSON
        with open(meta_file) as f:
            self.metas = json.load(f)
    
    def __getitem__(self, index):
        input = {}
        meta = self.metas[index]
        
        # 读取图像
        image = self.image_reader(meta['filename'])
        input.update({'image': image, 'filename': meta['filename']})
        
        # 读取标签（如果存在）
        if 'clsname' in meta:
            input['clsname'] = meta['clsname']
        if 'label' in meta:
            input['label'] = meta['label']
        
        # 读取掩码（测试时的ground truth）
        if not self.training and 'maskname' in meta:
            mask = self.image_reader(meta['maskname'], is_mask=True)
            input['mask'] = mask
        
        # 数据增强和预处理
        if self.transform:
            input = self.transform(input)
        
        return input
```

### 2.2 元数据JSON格式

**train.json**：
```json
[
  {
    "filename": "data/MVTec-AD/mvtec_anomaly_detection/bottle/train/good/000.png",
    "clsname": "bottle",
    "label": 0
  },
  {
    "filename": "data/MVTec-AD/mvtec_anomaly_detection/cable/train/good/000.png",
    "clsname": "cable",
    "label": 0
  }
]
```

**test.json**：
```json
[
  {
    "filename": "data/MVTec-AD/mvtec_anomaly_detection/bottle/test/good/000.png",
    "maskname": "data/MVTec-AD/mvtec_anomaly_detection/bottle/ground_truth/good/000_mask.png",
    "clsname": "bottle",
    "label": 0
  },
  {
    "filename": "data/MVTec-AD/mvtec_anomaly_detection/bottle/test/broken_large/000.png",
    "maskname": "data/MVTec-AD/mvtec_anomaly_detection/bottle/ground_truth/broken_large/000_mask.png",
    "clsname": "bottle",
    "label": 1
  }
]
```

### 2.3 图像读取

**文件位置**：`datasets/image_reader.py`

```python
class OpencvReader:
    def __init__(self, image_dir, color_mode='RGB'):
        self.image_dir = image_dir
        self.color_mode = color_mode
    
    def __call__(self, filename, is_mask=False):
        # 拼接完整路径
        path = os.path.join(self.image_dir, filename)
        
        # 读取图像
        if is_mask:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = img[:, :, None]  # (H, W, 1)
        else:
            img = cv2.imread(path)
            if self.color_mode == 'RGB':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
```

## 3. CIFAR-10 数据集

### 3.1 CIFARDataset 实现

**文件位置**：`datasets/cifar_dataset.py`

**核心代码**：
```python
class CIFARDataset(BaseDataset):
    def __init__(self, root_dir, training=True, normals=[0,1,2,3,4], ...):
        super().__init__()
        self.training = training
        self.normals = normals  # 正常类别
        
        # 加载CIFAR-10数据
        if training:
            data_list = ['data_batch_1', 'data_batch_2', 'data_batch_3',
                        'data_batch_4', 'data_batch_5']
        else:
            data_list = ['test_batch']
        
        self.data = []
        self.labels = []
        for batch in data_list:
            path = os.path.join(root_dir, 'cifar-10-batches-py', batch)
            with open(path, 'rb') as f:
                entry = pickle.load(f, encoding='bytes')
                self.data.append(entry[b'data'])
                self.labels.extend(entry[b'labels'])
        
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # (N, 32, 32, 3)
        
        # 筛选样本
        if training:
            # 训练时只使用正常类别
            indices = [i for i, label in enumerate(self.labels) 
                      if label in normals]
        else:
            # 测试时使用所有类别
            indices = list(range(len(self.labels)))
        
        self.data = self.data[indices]
        self.labels = [self.labels[i] for i in indices]
        
        # 设置标签（正常=0，异常=1）
        self.anomaly_labels = [0 if label in normals else 1 
                               for label in self.labels]
    
    def __getitem__(self, index):
        input = {}
        image = self.data[index]
        input.update({
            'image': image,
            'label': self.anomaly_labels[index],
            'clsname': f'class_{self.labels[index]}'
        })
        
        if self.transform:
            input = self.transform(input)
        
        return input
```

### 3.2 正常类别配置

**4种实验配置**：

| 配置 | 正常类别 | 类别名称 |
|------|---------|---------|
| 01234 | [0,1,2,3,4] | airplane, automobile, bird, cat, deer |
| 02468 | [0,2,4,6,8] | airplane, bird, deer, frog, ship |
| 13579 | [1,3,5,7,9] | automobile, cat, dog, horse, truck |
| 56789 | [5,6,7,8,9] | dog, frog, horse, ship, truck |

**配置示例**：
```yaml
# experiments/CIFAR-10/01234/config.yaml
dataset:
  type: cifar10
  train:
    root_dir: ../../../data/CIFAR-10/
  test:
    root_dir: ../../../data/CIFAR-10/
  normals: [0, 1, 2, 3, 4]
```

## 4. 数据预处理

### 4.1 Transform 流程

**文件位置**：`datasets/transforms.py`

**训练时Transform**：
```python
train_transform = Compose([
    Resize(size=(256, 256)),           # 缩放
    RandomCrop(size=(224, 224)),       # 随机裁剪
    ColorJitter(0.1, 0.1, 0.1, 0.1),  # 颜色抖动（可选）
    ToTensor(),                        # 转Tensor
    Normalize(mean, std)               # 归一化
])
```

**测试时Transform**：
```python
test_transform = Compose([
    Resize(size=(224, 224)),           # 固定尺寸
    ToTensor(),
    Normalize(mean, std)
])
```

### 4.2 核心Transform实现

**Resize**：
```python
class Resize:
    def __init__(self, size):
        self.size = size  # (H, W)
    
    def __call__(self, input):
        image = input['image']
        image = cv2.resize(image, self.size[::-1])  # OpenCV用(W, H)
        input['image'] = image
        
        # 如果有mask，也需要resize
        if 'mask' in input:
            mask = input['mask']
            mask = cv2.resize(mask, self.size[::-1])
            input['mask'] = mask
        
        return input
```

**ToTensor**：
```python
class ToTensor:
    def __call__(self, input):
        image = input['image']
        # (H, W, C) → (C, H, W)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        input['image'] = image
        
        if 'mask' in input:
            mask = input['mask']
            mask = torch.from_numpy(mask).float()
            input['mask'] = mask
        
        return input
```

**Normalize**：
```python
class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
    
    def __call__(self, input):
        image = input['image']
        image = (image / 255.0 - self.mean) / self.std
        input['image'] = image
        return input
```

### 4.3 归一化参数

**ImageNet标准**（默认）：
```python
pixel_mean = [0.485, 0.456, 0.406]
pixel_std = [0.229, 0.224, 0.225]
```

**配置**：
```yaml
dataset:
  pixel_mean: [0.485, 0.456, 0.406]
  pixel_std: [0.229, 0.224, 0.225]
```

## 5. DataLoader 构建

### 5.1 构建流程

**文件位置**：`datasets/data_builder.py`

```python
def build_dataloader(config, training=True, distributed=True):
    # 1. 构建数据集
    if config.dataset.type == 'custom':
        dataset = CustomDataset(
            image_reader=build_image_reader(config),
            meta_file=config.dataset.train.meta_file if training 
                     else config.dataset.test.meta_file,
            training=training,
            transform=build_transform(config, training)
        )
    elif config.dataset.type == 'cifar10':
        dataset = CIFARDataset(
            root_dir=config.dataset.train.root_dir if training
                    else config.dataset.test.root_dir,
            training=training,
            normals=config.dataset.normals,
            transform=build_transform(config, training)
        )
    
    # 2. 构建Sampler（分布式训练）
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=training)
    else:
        sampler = None
    
    # 3. 构建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config.dataset.batch_size,
        shuffle=(sampler is None and training),
        sampler=sampler,
        num_workers=config.dataset.workers,
        pin_memory=True,
        drop_last=training
    )
    
    return dataloader
```

### 5.2 配置参数

```yaml
dataset:
  type: custom  # 或 cifar10
  batch_size: 8
  workers: 4    # DataLoader线程数
  input_size: [224, 224]
  pixel_mean: [0.485, 0.456, 0.406]
  pixel_std: [0.229, 0.224, 0.225]
```

## 6. 数据增强（可选）

### 6.1 MVTec-AD 增强

**默认不使用增强**（工业场景变化小）：
```yaml
dataset:
  train:
    hflip: False    # 水平翻转
    vflip: False    # 垂直翻转
    rotate: False   # 旋转
```

**如需启用**：
```python
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, input):
        if random.random() < self.p:
            image = input['image']
            image = cv2.flip(image, 1)  # 水平翻转
            input['image'] = image
        return input
```

### 6.2 CIFAR-10 增强

**可选增强**：
```python
class RandomCrop:
    def __init__(self, size, padding=4):
        self.size = size
        self.padding = padding
    
    def __call__(self, input):
        image = input['image']
        # 先padding
        image = cv2.copyMakeBorder(
            image, self.padding, self.padding, 
            self.padding, self.padding,
            cv2.BORDER_REFLECT
        )
        # 随机裁剪
        h, w = image.shape[:2]
        top = random.randint(0, h - self.size[0])
        left = random.randint(0, w - self.size[1])
        image = image[top:top+self.size[0], left:left+self.size[1]]
        input['image'] = image
        return input
```

## 7. 批次数据格式

### 7.1 训练批次

```python
batch = {
    'image': torch.tensor((B, 3, 224, 224)),  # 图像
    'label': torch.tensor((B,)),               # 标签 (0=正常, 1=异常)
    'clsname': ['bottle', 'cable', ...],       # 类别名
    'filename': ['path1', 'path2', ...]        # 文件路径
}
```

### 7.2 测试批次

```python
batch = {
    'image': torch.tensor((B, 3, 224, 224)),
    'label': torch.tensor((B,)),
    'mask': torch.tensor((B, 1, 224, 224)),    # Ground truth掩码
    'clsname': ['bottle', ...],
    'filename': ['path1', ...]
}
```

## 8. 自定义数据集

### 8.1 准备数据

**目录结构**：
```
my_dataset/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
└── test/
    ├── class1/
    └── class2/
```

### 8.2 生成元数据JSON

```python
import json
import os
from glob import glob

def generate_meta_json(data_dir, output_file, is_train=True):
    metas = []
    
    split = 'train' if is_train else 'test'
    for clsname in os.listdir(os.path.join(data_dir, split)):
        cls_dir = os.path.join(data_dir, split, clsname)
        for img_path in glob(os.path.join(cls_dir, '*.jpg')):
            meta = {
                'filename': os.path.relpath(img_path, data_dir),
                'clsname': clsname,
                'label': 0  # 训练时全部为0
            }
            metas.append(meta)
    
    with open(output_file, 'w') as f:
        json.dump(metas, f, indent=2)

# 使用
generate_meta_json('my_dataset', 'train.json', is_train=True)
generate_meta_json('my_dataset', 'test.json', is_train=False)
```

### 8.3 修改配置文件

```yaml
dataset:
  type: custom
  image_reader:
    type: opencv
    kwargs:
      image_dir: ../../data/my_dataset/
      color_mode: RGB
  train:
    meta_file: ../../data/my_dataset/train.json
  test:
    meta_file: ../../data/my_dataset/test.json
```

## 9. 数据统计

### 9.1 查看数据集大小

```python
from datasets.data_builder import build_dataloader

config = load_config('config.yaml')
train_loader = build_dataloader(config, training=True, distributed=False)
test_loader = build_dataloader(config, training=False, distributed=False)

print(f'Train samples: {len(train_loader.dataset)}')
print(f'Test samples: {len(test_loader.dataset)}')
print(f'Train batches: {len(train_loader)}')
print(f'Test batches: {len(test_loader)}')
```

### 9.2 类别分布

```python
from collections import Counter

labels = [sample['label'] for sample in train_loader.dataset]
print(Counter(labels))
# {0: 3500, 1: 0}  # 训练时只有正常样本

labels = [sample['label'] for sample in test_loader.dataset]
print(Counter(labels))
# {0: 1200, 1: 800}  # 测试时有正常和异常
```

## 10. 调试技巧

### 10.1 可视化批次数据

```python
import matplotlib.pyplot as plt

def visualize_batch(batch):
    images = batch['image']  # (B, 3, H, W)
    labels = batch['label']
    
    # 反归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean
    images = images.clamp(0, 1)
    
    # 显示
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_title(f'Label: {labels[i].item()}')
            ax.axis('off')
    plt.show()

# 使用
for batch in train_loader:
    visualize_batch(batch)
    break
```

### 10.2 检查数据加载速度

```python
import time

start = time.time()
for i, batch in enumerate(train_loader):
    if i >= 100:
        break
elapsed = time.time() - start
print(f'Time per batch: {elapsed/100:.3f}s')
```

## 11. 常见问题

### Q1: 数据加载太慢怎么办？
A: 增加`workers`参数（如从4增到8），使用SSD存储数据，减小图像尺寸。

### Q2: 如何添加新的数据增强？
A: 在`datasets/transforms.py`中添加新的Transform类，然后在`build_transform`中注册。

### Q3: CIFAR-10为什么要指定normals？
A: 异常检测需要定义什么是"正常"，不同的normal配置对应不同的实验设置。

### Q4: 如何处理不同尺寸的图像？
A: 使用Resize transform统一到224×224，或在config中调整`input_size`。

## 12. 总结

UniAD数据处理的关键点：

1. **元数据驱动**：使用JSON文件管理数据集信息
2. **统一接口**：BaseDataset抽象公共逻辑
3. **灵活Transform**：支持多种数据增强
4. **分布式支持**：使用DistributedSampler
5. **异常检测特殊性**：训练只用正常样本

下一步：阅读 [训练流程](../training/TRAINING.md) 了解如何使用数据进行模型训练。
