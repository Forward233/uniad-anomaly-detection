# UniAD 进阶实践

本文档介绍如何自定义开发、调试技巧和性能优化。

## 1. 自定义数据集

### 1.1 准备数据

**目录结构**：
```
my_dataset/
├── train/
│   └── good/              # 只包含正常样本
│       ├── 001.jpg
│       ├── 002.jpg
│       └── ...
└── test/
    ├── good/              # 正常样本
    │   ├── 001.jpg
    │   └── ...
    └── defect_type1/      # 异常样本
        ├── 001.jpg
        ├── 001_mask.png   # ground truth掩码
        └── ...
```

### 1.2 生成元数据JSON

```python
import json
import os
from pathlib import Path

def generate_train_json(data_dir, output_file):
    metas = []
    train_dir = Path(data_dir) / 'train' / 'good'
    
    for img_path in sorted(train_dir.glob('*.jpg')):
        meta = {
            'filename': str(img_path.relative_to(data_dir)),
            'clsname': 'my_class',
            'label': 0
        }
        metas.append(meta)
    
    with open(output_file, 'w') as f:
        json.dump(metas, f, indent=2)
    
    print(f'Generated {len(metas)} training samples')

def generate_test_json(data_dir, output_file):
    metas = []
    test_dir = Path(data_dir) / 'test'
    
    for defect_type in sorted(test_dir.iterdir()):
        if not defect_type.is_dir():
            continue
        
        is_good = (defect_type.name == 'good')
        
        for img_path in sorted(defect_type.glob('*.jpg')):
            mask_path = img_path.with_name(f'{img_path.stem}_mask.png')
            
            meta = {
                'filename': str(img_path.relative_to(data_dir)),
                'clsname': 'my_class',
                'label': 0 if is_good else 1
            }
            
            if mask_path.exists():
                meta['maskname'] = str(mask_path.relative_to(data_dir))
            
            metas.append(meta)
    
    with open(output_file, 'w') as f:
        json.dump(metas, f, indent=2)
    
    print(f'Generated {len(metas)} test samples')

# 使用
generate_train_json('data/my_dataset', 'data/my_dataset/train.json')
generate_test_json('data/my_dataset', 'data/my_dataset/test.json')
```

### 1.3 修改配置文件

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
    rebalance: False
    hflip: False
    vflip: False
    rotate: False
  
  test:
    meta_file: ../../data/my_dataset/test.json
  
  input_size: [224, 224]
  pixel_mean: [0.485, 0.456, 0.406]
  pixel_std: [0.229, 0.224, 0.225]
  batch_size: 8
  workers: 4
```

### 1.4 训练和评估

```bash
cd experiments/my_experiment
bash train_torch.sh 1 0
bash eval_torch.sh 1 0
```

## 2. 修改网络结构

### 2.1 更换Backbone

**使用ResNet50**：
```yaml
net:
  - name: backbone
    type: models.backbones.resnet50
    frozen: True
    kwargs:
      pretrained: True
      outlayers: [2, 3, 4]  # 选择层
```

**使用EfficientNet-B0**（更小更快）：
```yaml
net:
  - name: backbone
    type: models.backbones.efficientnet_b0
    frozen: True
    kwargs:
      pretrained: True
      outlayers: [1, 2, 3, 4]
```

### 2.2 调整Transformer参数

```yaml
net:
  - name: reconstruction
    type: models.reconstructions.UniAD
    kwargs:
      hidden_dim: 512        # 增大隐藏维度
      nhead: 16              # 增加注意力头数
      num_encoder_layers: 6  # 增加编码器层数
      num_decoder_layers: 6  # 增加解码器层数
      dim_feedforward: 2048  # 增大前馈网络维度
```

**权衡**：
- 更大的模型 = 更强的表达能力，但训练更慢、需要更多显存
- 更小的模型 = 训练更快，但可能性能略差

### 2.3 添加新的损失函数

**实现L1损失**：
```python
# utils/criterion_helper.py
class FeatureL1Loss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.l1 = nn.L1Loss()
    
    def forward(self, outputs):
        feature = outputs['feature']
        reconstructed = outputs['reconstructed']
        loss = self.l1(reconstructed, feature)
        return {'FeatureL1Loss': loss * self.weight}

# 注册
def build_criterion(criterion_cfg):
    criterion_map = {
        'FeatureMSELoss': FeatureMSELoss,
        'FeatureL1Loss': FeatureL1Loss,  # 添加新损失
    }
    # ...
```

**配置**：
```yaml
criterion:
  - name: FeatureMSELoss
    type: FeatureMSELoss
    kwargs:
      weight: 1.0
  - name: FeatureL1Loss
    type: FeatureL1Loss
    kwargs:
      weight: 0.1
```

### 2.4 自定义Transform

```python
# datasets/transforms.py
class MyCustomTransform:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    
    def __call__(self, input):
        image = input['image']
        # 自定义处理
        image = self.process(image)
        input['image'] = image
        return input
    
    def process(self, image):
        # 实现你的转换逻辑
        return image

# 在build_transform中注册
def build_transform(config, training):
    transforms = [
        Resize(config.dataset.input_size),
        MyCustomTransform(param1=..., param2=...),  # 添加
        ToTensor(),
        Normalize(config.dataset.pixel_mean, config.dataset.pixel_std)
    ]
    return Compose(transforms)
```

## 3. 调试技巧

### 3.1 使用IDE断点调试

**VSCode配置**（`.vscode/launch.json`）：
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Train UniAD",
            "type": "python",
            "request": "launch",
            "program": "tools/train_val.py",
            "args": [
                "--config", "experiments/MVTec-AD/config.yaml"
            ],
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}",
                "CUDA_VISIBLE_DEVICES": "0"
            },
            "console": "integratedTerminal"
        }
    ]
}
```

### 3.2 打印中间变量

```python
# 在forward函数中添加
def forward(self, x):
    feat = self.backbone(x)
    print(f'Backbone output shape: {[f.shape for f in feat]}')
    
    fused = self.neck(feat)
    print(f'Neck output shape: {fused[0].shape}')
    
    rec = self.reconstruction(fused[0])
    print(f'Reconstruction output shape: {rec.shape}')
    
    return {'reconstructed': rec, 'feature': fused[0]}
```

### 3.3 可视化梯度

```python
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f'{name}: grad_norm={grad_norm:.6f}')
            
            if grad_norm > 100:
                print(f'  WARNING: Large gradient!')
            elif grad_norm < 1e-6:
                print(f'  WARNING: Very small gradient!')
```

### 3.4 检查数据分布

```python
def check_data_statistics(dataloader):
    all_images = []
    all_labels = []
    
    for batch in dataloader:
        all_images.append(batch['image'])
        all_labels.extend(batch['label'].numpy())
    
    all_images = torch.cat(all_images, dim=0)
    
    print(f'Image shape: {all_images.shape}')
    print(f'Image mean: {all_images.mean().item():.4f}')
    print(f'Image std: {all_images.std().item():.4f}')
    print(f'Image min: {all_images.min().item():.4f}')
    print(f'Image max: {all_images.max().item():.4f}')
    print(f'Label distribution: {Counter(all_labels)}')
```

### 3.5 单样本调试

```python
def debug_single_sample(model, image):
    model.eval()
    
    # 添加batch维度
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    image = image.cuda()
    
    # 逐步执行
    print('1. Backbone...')
    feat = model.backbone(image)
    print(f'   Output shapes: {[f.shape for f in feat]}')
    
    print('2. Neck...')
    fused = model.neck(feat)
    print(f'   Output shape: {fused[0].shape}')
    
    print('3. Reconstruction...')
    rec = model.reconstruction(fused[0])
    print(f'   Output shape: {rec.shape}')
    
    print('4. Compute loss...')
    loss = F.mse_loss(rec, fused[0])
    print(f'   Loss: {loss.item():.4f}')
```

## 4. 性能优化

### 4.1 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(max_epoch):
    for batch in train_loader:
        images = batch['image'].cuda()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

**效果**：
- 训练速度提升约30-50%
- 显存占用减少约40%
- 精度基本不变

### 4.2 梯度累积

```python
accumulation_steps = 4

for i, batch in enumerate(train_loader):
    images = batch['image'].cuda()
    outputs = model(images)
    loss = criterion(outputs) / accumulation_steps
    
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**适用场景**：
- 显存不足，无法使用大batch_size
- 等效于batch_size增大4倍

### 4.3 DataLoader优化

```yaml
dataset:
  workers: 8              # 增加线程数
  pin_memory: True        # 使用pinned memory
  prefetch_factor: 2      # 预取批次数
```

```python
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True  # 保持worker进程
)
```

### 4.4 模型编译（PyTorch 2.0+）

```python
import torch

model = torch.compile(model)  # JIT编译加速
```

### 4.5 推理加速

**TorchScript导出**：
```python
model.eval()
example_input = torch.randn(1, 3, 224, 224).cuda()
traced_model = torch.jit.trace(model, example_input)
traced_model.save('model_traced.pt')

# 加载和使用
model = torch.jit.load('model_traced.pt')
output = model(input)
```

**ONNX导出**：
```python
torch.onnx.export(
    model,
    example_input,
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

## 5. 超参数调优

### 5.1 学习率

**尝试范围**：
- 0.00001 - 0.001
- 使用学习率查找器

```python
from torch_lr_finder import LRFinder

lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
lr_finder.plot()
suggested_lr = lr_finder.suggestion()
```

### 5.2 Batch Size

**经验法则**：
- 尽可能大（在显存允许的范围内）
- MVTec-AD: 8-16
- CIFAR-10: 16-32

### 5.3 Feature Jitter

**调整scale**：
```yaml
net:
  - name: reconstruction
    kwargs:
      feature_jitter:
        scale: 10.0    # 减小：更稳定，可能欠拟合
        # scale: 30.0  # 增大：更鲁棒，可能过拟合
        prob: 1.0
```

### 5.4 Neighbor Mask

**调整邻域大小**：
```yaml
net:
  - name: reconstruction
    kwargs:
      neighbor_mask:
        neighbor_size: [5, 5]    # 更小：更局部
        # neighbor_size: [9, 9]  # 更大：更全局
        mask: [True, True, True]
```

## 6. 实验管理

### 6.1 使用Weights & Biases

```python
import wandb

wandb.init(project='uniad', name='experiment_name')
wandb.config.update(config)

for epoch in range(max_epoch):
    loss = train_one_epoch(...)
    metrics = validate(...)
    
    wandb.log({
        'train/loss': loss,
        'val/pixel_auc': metrics['pixel_auc'],
        'val/image_auc': metrics['image_auc'],
        'epoch': epoch
    })
```

### 6.2 实验配置管理

**目录结构**：
```
experiments/
├── my_exp_v1/
│   ├── config.yaml
│   ├── checkpoints/
│   └── log/
├── my_exp_v2/
│   ├── config.yaml
│   ├── checkpoints/
│   └── log/
└── ...
```

**命名规范**：
- `baseline`: 基础配置
- `baseline_lr001`: 修改学习率
- `baseline_b4`: 使用EfficientNet-B4
- `baseline_aug`: 添加数据增强

### 6.3 结果对比

```python
import pandas as pd

results = {
    'Experiment': ['baseline', 'baseline_lr001', 'baseline_b4'],
    'Pixel AUC': [98.1, 98.3, 98.5],
    'Image AUC': [98.7, 98.9, 99.1],
    'Training Time (h)': [12, 14, 15]
}

df = pd.DataFrame(results)
print(df.to_markdown(index=False))
```

## 7. 部署相关

### 7.1 模型量化

```python
# 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 静态量化
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
# 校准
with torch.no_grad():
    for batch in calibration_loader:
        model(batch['image'])
torch.quantization.convert(model, inplace=True)
```

### 7.2 模型剪枝

```python
import torch.nn.utils.prune as prune

# 剪枝Transformer的线性层
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)

# 永久移除剪枝
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        prune.remove(module, 'weight')
```

### 7.3 模型蒸馏

```python
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_output, teacher_output, target):
        # 蒸馏损失
        distill_loss = self.kl_div(
            F.log_softmax(student_output / self.temperature, dim=1),
            F.softmax(teacher_output / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        
        # 原始损失
        student_loss = F.mse_loss(student_output, target)
        
        # 组合
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * student_loss
        return total_loss
```

## 8. 常见问题解决

### 8.1 训练不收敛

**检查清单**：
1. 学习率是否过大/过小
2. 数据是否正确归一化
3. Backbone是否正确加载预训练权重
4. 损失函数是否正确
5. 梯度是否正常

**解决方法**：
```python
# 1. 降低学习率
lr = 0.00001

# 2. 检查数据归一化
mean, std = compute_dataset_stats(train_loader)

# 3. 验证预训练权重
check_pretrained_weights(model.backbone)

# 4. 打印损失
print(f'Loss: {loss.item()}')

# 5. 检查梯度
check_gradients(model)
```

### 8.2 显存溢出

**解决方法**：
```yaml
# 1. 减小batch_size
dataset:
  batch_size: 4

# 2. 减小输入尺寸
dataset:
  input_size: [192, 192]

# 3. 使用梯度累积
accumulation_steps: 4

# 4. 减小模型
net:
  - name: reconstruction
    kwargs:
      hidden_dim: 128
      num_encoder_layers: 2
      num_decoder_layers: 2
```

### 8.3 过拟合

**解决方法**：
```yaml
# 1. 增加weight_decay
trainer:
  optimizer:
    kwargs:
      weight_decay: 0.001

# 2. 增大Feature Jitter
net:
  - name: reconstruction
    kwargs:
      feature_jitter:
        scale: 30.0

# 3. 添加Dropout
net:
  - name: reconstruction
    kwargs:
      dropout: 0.2

# 4. 数据增强
dataset:
  train:
    hflip: True
    vflip: True
```

### 8.4 推理速度慢

**优化方法**：
1. 使用TorchScript
2. 使用混合精度
3. 批量推理
4. 模型量化
5. 更小的Backbone

## 9. 进阶话题

### 9.1 多任务学习

同时训练分类和重建：
```python
criterion = {
    'reconstruction': FeatureMSELoss(weight=1.0),
    'classification': nn.CrossEntropyLoss()
}
```

### 9.2 域适应

源域（MVTec-AD）→ 目标域（自定义数据）：
```python
# 1. 在MVTec-AD预训练
# 2. 在目标域微调
for param in model.backbone.parameters():
    param.requires_grad = False  # 冻结backbone
# 只训练reconstruction
```

### 9.3 半监督学习

利用未标注数据：
```python
# 使用伪标签
with torch.no_grad():
    pseudo_labels = model(unlabeled_data)
    confident = (pseudo_labels.max(dim=1) > threshold)
    
# 在confident样本上训练
```

## 10. 最佳实践

1. **先小规模实验**：在小数据集上验证想法
2. **版本控制**：使用git管理代码和配置
3. **记录实验**：详细记录每次实验的配置和结果
4. **可复现性**：固定随机种子，记录环境版本
5. **渐进式开发**：一次只改一个变量
6. **性能监控**：使用tensorboard或wandb
7. **代码审查**：检查代码质量和效率
8. **文档更新**：及时更新文档和注释

## 11. 参考资源

- **原始论文**: https://arxiv.org/abs/2206.03687
- **PyTorch官方文档**: https://pytorch.org/docs/
- **异常检测综述**: https://github.com/hoya012/awesome-anomaly-detection
- **Transformer教程**: https://jalammar.github.io/illustrated-transformer/

## 12. 总结

进阶开发的关键点：

1. **自定义数据集**：准备JSON元数据，修改配置
2. **修改网络**：更换Backbone、调整Transformer参数
3. **调试技巧**：使用断点、打印变量、可视化
4. **性能优化**：混合精度、梯度累积、DataLoader优化
5. **超参数调优**：学习率、batch_size、jitter scale
6. **实验管理**：使用wandb、规范命名、记录结果
7. **部署优化**：量化、剪枝、蒸馏

通过本文档的学习，你应该能够：
- 使用自己的数据集训练UniAD
- 修改网络结构和参数
- 调试和优化模型
- 部署到生产环境

祝你在异常检测领域取得成功！
