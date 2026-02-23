# UniAD 整体架构

本文档详细介绍 UniAD 项目的整体架构设计、三层结构和数据流程。

## 1. 架构概览

UniAD 采用经典的**特征提取-重建**范式进行异常检测，整体架构分为三个主要层次：

```
输入图像 → Backbone → Neck → Reconstruction → 重建特征 → 异常评分
```

### 1.1 三层架构

```
┌─────────────┐
│  输入图像    │ (H×W×3)
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│  Backbone              │
│  - ResNet              │
│  - EfficientNet        │
└──────┬──────────────────┘
       │ 多尺度特征
       ▼
┌─────────────────────────┐
│  Neck (MFCN)           │
│  - 特征融合             │
│  - 通道调整             │
└──────┬──────────────────┘
       │ 融合特征
       ▼
┌─────────────────────────┐
│  Reconstruction        │
│  - Transformer         │
│  - Feature Jitter      │
│  - Neighbor Mask       │
└──────┬──────────────────┘
       │ 重建特征
       ▼
┌─────────────────────────┐
│  Loss & Score          │
│  - MSE Loss (训练)     │
│  - Anomaly Score (测试)│
└─────────────────────────┘
```

### 1.2 核心思想

**正常样本重建假设**：
- 模型在训练时只见过正常样本
- 学会重建正常样本的特征
- 对异常样本，重建误差会变大
- 重建误差 = 异常分数

## 2. 详细架构

### 2.1 Backbone（骨干网络）

**作用**：提取图像的深层特征表示

**支持的网络**：
- **ResNet**: ResNet18/34/50/101/152
- **EfficientNet**: EfficientNet-B0/B1/B2/B3/B4

**输出**：多尺度特征图
```python
# 以EfficientNet-B4为例
outlayers = [1, 2, 3, 4]  # 选择第1,2,3,4层特征
outstrides = [2, 4, 8, 16]  # 对应的下采样倍数
```

**代码位置**：
- `models/backbones/resnet.py`
- `models/backbones/efficientnet/`

### 2.2 Neck（特征融合）

**作用**：融合多尺度特征，统一特征维度

**实现**：MFCN (Multi-scale Feature Concatenation Network)

**处理流程**：
```python
# 伪代码
multi_scale_features = [feat1, feat2, feat3, feat4]  # 来自Backbone
# 上采样到统一尺度
upsampled = [upsample(f, target_size) for f in multi_scale_features]
# 拼接
concatenated = concat(upsampled, dim=channel)
# 降维
fused = conv1x1(concatenated, out_channels)
```

**输出**：统一尺度的融合特征

**代码位置**：`models/necks/mfcn.py`

### 2.3 Reconstruction（重建网络）

**作用**：使用Transformer重建特征

**核心组件**：
1. **位置编码**：为特征添加空间位置信息
2. **Transformer Encoder**：编码输入特征
3. **Transformer Decoder**：解码重建特征
4. **Feature Jitter**：训练时添加特征扰动
5. **Neighbor Mask**：限制自注意力范围

**代码位置**：`models/reconstructions/uniad.py`

## 3. 数据流程

### 3.1 训练流程

```
正常样本图像
    ↓
Backbone提取特征
    ↓
Neck融合特征
    ↓
添加Feature Jitter (随机扰动)
    ↓
Transformer重建
    ↓
计算MSE Loss (原始特征 vs 重建特征)
    ↓
反向传播更新参数
```

**关键点**：
- 只使用正常样本训练
- Feature Jitter增强鲁棒性
- MSE Loss最小化重建误差

### 3.2 测试流程

```
测试样本图像
    ↓
Backbone提取特征
    ↓
Neck融合特征
    ↓
Transformer重建 (无Jitter)
    ↓
计算重建误差
    ↓
上采样到原图尺寸
    ↓
异常热图 (像素级异常分数)
```

**关键点**：
- 测试时不添加Jitter
- 重建误差映射为异常分数
- 支持像素级异常定位

### 3.3 特征维度变化

以MVTec-AD配置为例：

```
输入图像: (Batch, 3, 224, 224)
    ↓
Backbone (EfficientNet-B4, outlayer=4):
    layer1: (Batch, 32, 112, 112)   # stride=2
    layer2: (Batch, 56, 56, 56)     # stride=4
    layer3: (Batch, 160, 28, 28)    # stride=8
    layer4: (Batch, 272, 14, 14)    # stride=16
    ↓
Neck (MFCN, outstride=16):
    上采样统一到14×14
    拼接: (Batch, 520, 14, 14)
    降维: (Batch, 272, 14, 14)
    ↓
Reconstruction (UniAD):
    展平: (Batch, 196, 272)  # 196=14×14
    Transformer: (Batch, 196, 272)
    重塑: (Batch, 272, 14, 14)
    ↓
上采样到原图:
    (Batch, 272, 224, 224)
```

## 4. 配置映射

### 4.1 配置文件结构

```yaml
net:
  - name: backbone
    type: models.backbones.efficientnet_b4
    kwargs:
      pretrained: True
      outlayers: [1, 2, 3, 4]
  
  - name: neck
    prev: backbone
    type: models.necks.MFCN
    kwargs:
      outstrides: [16]
  
  - name: reconstruction
    prev: neck
    type: models.reconstructions.UniAD
    kwargs:
      hidden_dim: 256
      nhead: 8
      num_encoder_layers: 4
      num_decoder_layers: 4
```

### 4.2 代码实现映射

**配置解析** → `utils/misc_helper.py: update_config()`
```python
# 自动计算特征维度
config.net[i].kwargs.inplanes = [...]  # 输入通道数
config.net[i].kwargs.instrides = [...]  # 输入步长
```

**模型构建** → `models/model_helper.py: ModelHelper`
```python
class ModelHelper(nn.Module):
    def build_models(self):
        # 依次构建backbone、neck、reconstruction
        for layer_cfg in config.net:
            layer = build_layer(layer_cfg)
            self.add_module(layer_cfg.name, layer)
```

## 5. 关键设计

### 5.1 为什么使用预训练Backbone？

**优点**：
- 丰富的特征表示能力
- 无需大量异常检测数据训练
- 迁移学习效果好

**实现**：
```python
# models/backbones/efficientnet/utils.py
def load_pretrained_weights(model, model_name):
    # 从ImageNet预训练模型加载权重
    pretrained_dict = load_url(url_map[model_name])
    model.load_state_dict(pretrained_dict)
```

### 5.2 为什么冻结Backbone？

**配置**：
```yaml
frozen_layers: [backbone]
```

**原因**：
- 预训练特征已经足够好
- 减少训练参数，防止过拟合
- 加快训练速度

**实现**：
```python
# tools/train_val.py
for name, param in model.named_parameters():
    if any(layer in name for layer in frozen_layers):
        param.requires_grad = False
```

### 5.3 为什么使用Transformer？

**优势**：
- 全局感受野，捕捉长距离依赖
- 自注意力机制，灵活建模特征关系
- 位置编码，保留空间信息

**实现**：
```python
# models/reconstructions/uniad.py
class Transformer(nn.Module):
    def __init__(self, hidden_dim, feature_size, ...):
        self.encoder = TransformerEncoder(...)
        self.decoder = TransformerDecoder(...)
```

## 6. 模型初始化

### 6.1 初始化流程

```python
# tools/train_val.py
def main():
    # 1. 构建模型
    model = ModelHelper(config)
    
    # 2. 加载预训练Backbone
    # 自动从ImageNet加载
    
    # 3. 初始化其他层
    # Xavier uniform初始化
    
    # 4. 冻结指定层
    for layer in frozen_layers:
        freeze(layer)
    
    # 5. 移动到GPU
    model = model.cuda()
    
    # 6. 包装为DDP (分布式)
    model = DDP(model)
```

### 6.2 初始化方法

**配置**：
```yaml
net:
  - name: reconstruction
    kwargs:
      initializer:
        method: xavier_uniform
```

**实现**：
```python
# models/initializer.py
def initialize_from_cfg(model, cfg):
    if cfg.method == 'xavier_uniform':
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
```

## 7. 模型保存和加载

### 7.1 Checkpoint结构

```python
checkpoint = {
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'best_metric': best_metric,
    'config': config
}
```

### 7.2 保存策略

**配置**：
```yaml
saver:
  always_save: False  # 只保存最优模型
  save_dir: checkpoints/
```

**实现**：
```python
# utils/misc_helper.py
def save_checkpoint(state, is_best, save_path):
    if is_best:
        torch.save(state, os.path.join(save_path, 'ckpt.pth.tar'))
```

### 7.3 加载流程

```python
# utils/misc_helper.py
def load_state(path, model, optimizer=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']
```

## 8. 总结

UniAD架构的核心特点：

1. **模块化设计**：Backbone-Neck-Reconstruction清晰分离
2. **预训练迁移**：利用ImageNet预训练特征
3. **Transformer重建**：灵活建模特征关系
4. **端到端训练**：只需正常样本即可训练
5. **像素级检测**：支持异常区域定位

下一步：阅读 [模型详解](MODELS.md) 了解各模块的具体实现细节。
