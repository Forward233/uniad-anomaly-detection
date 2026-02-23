# UniAD 模型详解

本文档深入解析 UniAD 的三个核心模块：Backbone、Neck、Reconstruction。

## 1. Backbone 骨干网络

### 1.1 ResNet

**文件位置**：`models/backbones/resnet.py`

**核心实现**：
```python
class ResNet(nn.Module):
    def __init__(self, block, layers, outlayers=[1,2,3]):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 构建4个stage
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.outlayers = outlayers
```

**特征提取**：
```python
def forward(self, x):
    x = self.conv1(x)  # stride=2
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)  # stride=2，总stride=4
    
    outs = []
    if 1 in self.outlayers:
        outs.append(self.layer1(x))  # stride=4
    if 2 in self.outlayers:
        outs.append(self.layer2(x))  # stride=8
    if 3 in self.outlayers:
        outs.append(self.layer3(x))  # stride=16
    if 4 in self.outlayers:
        outs.append(self.layer4(x))  # stride=32
    
    return outs
```

**支持的变体**：
- ResNet18: layers=[2, 2, 2, 2]
- ResNet34: layers=[3, 4, 6, 3]
- ResNet50: layers=[3, 4, 6, 3] + Bottleneck
- ResNet101: layers=[3, 4, 23, 3] + Bottleneck
- ResNet152: layers=[3, 8, 36, 3] + Bottleneck

### 1.2 EfficientNet

**文件位置**：`models/backbones/efficientnet/`

**核心实现**：
```python
class EfficientNet(nn.Module):
    def __init__(self, blocks_args, global_params, outlayers=[1,2,3,4,5]):
        super().__init__()
        # Stem
        self._conv_stem = Conv2dStaticSamePadding(3, 32, kernel_size=3, stride=2)
        self._bn0 = nn.BatchNorm2d(32)
        
        # Blocks
        self._blocks = nn.ModuleList([])
        for block_args in blocks_args:
            self._blocks.append(MBConvBlock(block_args, global_params))
        
        self.outlayers = outlayers
```

**特征提取**：
```python
def forward(self, x):
    x = self._swish(self._bn0(self._conv_stem(x)))
    
    outs = []
    for idx, block in enumerate(self._blocks):
        x = block(x)
        if idx + 1 in self.outlayers:
            outs.append(x)
    
    return outs
```

**EfficientNet-B4特征维度**：
```python
outlayers = [1, 2, 3, 4]
# layer1: (B, 32, 112, 112), stride=2
# layer2: (B, 56, 56, 56), stride=4  
# layer3: (B, 160, 28, 28), stride=8
# layer4: (B, 272, 14, 14), stride=16
```

**预训练加载**：
```python
# models/backbones/efficientnet/utils.py
def load_pretrained_weights(model, model_name):
    pretrained_dict = load_url(url_map[model_name])
    model.load_state_dict(pretrained_dict)
```

## 2. Neck 特征融合

### 2.1 MFCN (Multi-scale Feature Concatenation Network)

**文件位置**：`models/necks/mfcn.py`

**核心思想**：
- 将多尺度特征上采样到统一尺度
- 拼接后降维，得到融合特征

**实现**：
```python
class MFCN(nn.Module):
    def __init__(self, inplanes, outplanes, instrides, outstrides):
        super().__init__()
        # inplanes: [32, 56, 160, 272] (EfficientNet-B4)
        # instrides: [2, 4, 8, 16]
        # outstrides: [16]
        
        # 上采样比例
        self.scales = [instride // outstride for instride in instrides]
        # [2//16, 4//16, 8//16, 16//16] = [8, 4, 2, 1]
        
        # 降维卷积
        concat_channels = sum(inplanes)  # 32+56+160+272=520
        self.conv = nn.Conv2d(concat_channels, outplanes[0], kernel_size=1)
    
    def forward(self, features):
        # features: 来自Backbone的多尺度特征
        target_size = features[-1].shape[2:]  # 最低分辨率的尺寸
        
        # 上采样到统一尺度
        upsampled = []
        for feat, scale in zip(features, self.scales):
            if scale > 1:
                feat = F.interpolate(feat, scale_factor=scale, mode='bilinear')
            upsampled.append(feat)
        
        # 拼接
        concat_feat = torch.cat(upsampled, dim=1)
        
        # 降维
        out = self.conv(concat_feat)
        
        return [out]
```

**示例**：
```python
# 输入：EfficientNet-B4的4个特征
features = [
    torch.randn(8, 32, 112, 112),   # layer1, stride=2
    torch.randn(8, 56, 56, 56),     # layer2, stride=4
    torch.randn(8, 160, 28, 28),    # layer3, stride=8
    torch.randn(8, 272, 14, 14)     # layer4, stride=16
]

# MFCN处理
mfcn = MFCN(
    inplanes=[32, 56, 160, 272],
    outplanes=[272],
    instrides=[2, 4, 8, 16],
    outstrides=[16]
)

# 输出：融合特征
out = mfcn(features)  # [(8, 272, 14, 14)]
```

## 3. Reconstruction 重建网络

### 3.1 UniAD 核心模块

**文件位置**：`models/reconstructions/uniad.py`

**整体结构**：
```python
class UniAD(nn.Module):
    def __init__(self, inplanes, instrides, feature_size, hidden_dim, ...):
        super().__init__()
        # 输入投影
        self.input_proj = nn.Linear(inplanes[0], hidden_dim)
        
        # 位置编码
        self.pos_embed = build_position_embedding(...)
        
        # Transformer
        self.transformer = Transformer(hidden_dim, feature_size, ...)
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, inplanes[0])
        
        # 上采样
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=instrides[0])
```

**前向传播**：
```python
def forward(self, features):
    # features: (B, C, H, W)
    B, C, H, W = features.shape
    
    # 1. 展平: (B, C, H, W) → (B, H*W, C)
    feature_tokens = rearrange(features, 'b c h w -> b (h w) c')
    
    # 2. 添加Feature Jitter (训练时)
    if self.training:
        feature_tokens = self.add_jitter(feature_tokens)
    
    # 3. 投影到hidden_dim
    feature_tokens = self.input_proj(feature_tokens)
    
    # 4. 添加位置编码
    pos_embed = self.pos_embed(B)  # (B, H*W, hidden_dim)
    
    # 5. Transformer重建
    rec_tokens = self.transformer(feature_tokens, pos_embed)
    
    # 6. 投影回原始维度
    rec_tokens = self.output_proj(rec_tokens)
    
    # 7. 重塑: (B, H*W, C) → (B, C, H, W)
    rec_features = rearrange(rec_tokens, 'b (h w) c -> b c h w', h=H, w=W)
    
    # 8. 上采样到输入图像尺寸
    rec_features = self.upsample(rec_features)
    
    return rec_features
```

### 3.2 Feature Jitter

**原理**：训练时对特征添加随机扰动，增强模型鲁棒性

**实现**：
```python
def add_jitter(self, feature_tokens, scale=20.0, prob=1.0):
    # scale: 扰动强度
    # prob: 扰动概率
    
    if random.random() > prob:
        return feature_tokens
    
    # 随机扰动
    jitter = torch.randn_like(feature_tokens) * scale
    feature_tokens = feature_tokens + jitter
    
    return feature_tokens
```

**配置**：
```yaml
net:
  - name: reconstruction
    kwargs:
      feature_jitter:
        scale: 20.0
        prob: 1.0
```

**作用**：
- 防止模型过拟合到精确的特征值
- 提升对小扰动的鲁棒性
- 类似Dropout的正则化效果

### 3.3 Transformer 模块

**结构**：
```python
class Transformer(nn.Module):
    def __init__(self, hidden_dim, feature_size, neighbor_mask, ...):
        super().__init__()
        # Encoder
        self.encoder = TransformerEncoder(
            d_model=hidden_dim,
            nhead=nhead,
            num_layers=num_encoder_layers
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            d_model=hidden_dim,
            nhead=nhead,
            num_layers=num_decoder_layers
        )
        
        # Queries (可学习的查询向量)
        self.query_embed = nn.Embedding(feature_size[0]*feature_size[1], hidden_dim)
        
        # Neighbor Mask
        self.neighbor_mask = self._build_neighbor_mask(feature_size, neighbor_mask)
```

**前向传播**：
```python
def forward(self, feature_tokens, pos_embed):
    # 1. Encoder编码
    memory = self.encoder(
        src=feature_tokens + pos_embed,
        mask=self.neighbor_mask['enc']
    )
    
    # 2. Queries
    queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
    
    # 3. Decoder解码
    output = self.decoder(
        tgt=queries + pos_embed,
        memory=memory,
        tgt_mask=self.neighbor_mask['dec1'],
        memory_mask=self.neighbor_mask['dec2']
    )
    
    return output
```

### 3.4 Neighbor Mask

**原理**：限制自注意力的邻域范围，保持局部性

**实现**：
```python
def _build_neighbor_mask(self, feature_size, neighbor_mask_cfg):
    H, W = feature_size  # (14, 14)
    neighbor_size = neighbor_mask_cfg['neighbor_size']  # [7, 7]
    
    # 构建邻域掩码矩阵
    mask = torch.ones(H*W, H*W)  # (196, 196)
    
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            # 计算邻域范围
            h_start = max(0, i - neighbor_size[0] // 2)
            h_end = min(H, i + neighbor_size[0] // 2 + 1)
            w_start = max(0, j - neighbor_size[1] // 2)
            w_end = min(W, j + neighbor_size[1] // 2 + 1)
            
            # 设置邻域内的mask为0 (允许注意力)
            for ni in range(h_start, h_end):
                for nj in range(w_start, w_end):
                    nidx = ni * W + nj
                    mask[idx, nidx] = 0
    
    # 0表示允许注意力，1表示屏蔽
    mask = mask.bool()
    return mask
```

**配置**：
```yaml
net:
  - name: reconstruction
    kwargs:
      neighbor_mask:
        neighbor_size: [7, 7]
        mask: [True, True, True]  # 是否在[enc, dec1, dec2]使用mask
```

**示例**：
```
原图14×14，neighbor_size=[7,7]

位置(7,7)可以关注的邻域：
┌─────────────┐
│ . . . . . . │
│ . . . . . . │
│ . . [7,7] . │  7×7邻域
│ . . . . . . │
│ . . . . . . │
└─────────────┘
```

**作用**：
- 保持特征的局部性
- 减少计算复杂度
- 防止过度全局化

### 3.5 位置编码

**类型1：Learned Position Embedding**
```python
class LearnedPositionEmbedding(nn.Module):
    def __init__(self, feature_size, hidden_dim):
        super().__init__()
        H, W = feature_size
        self.row_embed = nn.Embedding(H, hidden_dim // 2)
        self.col_embed = nn.Embedding(W, hidden_dim // 2)
    
    def forward(self, B):
        H, W = self.row_embed.num_embeddings, self.col_embed.num_embeddings
        i = torch.arange(H, device=self.row_embed.weight.device)
        j = torch.arange(W, device=self.col_embed.weight.device)
        
        # 行编码
        row_emb = self.row_embed(i)  # (H, hidden_dim//2)
        row_emb = row_emb.unsqueeze(1).repeat(1, W, 1)  # (H, W, hidden_dim//2)
        
        # 列编码
        col_emb = self.col_embed(j)  # (W, hidden_dim//2)
        col_emb = col_emb.unsqueeze(0).repeat(H, 1, 1)  # (H, W, hidden_dim//2)
        
        # 拼接
        pos_emb = torch.cat([row_emb, col_emb], dim=-1)  # (H, W, hidden_dim)
        pos_emb = rearrange(pos_emb, 'h w c -> (h w) c')  # (H*W, hidden_dim)
        pos_emb = pos_emb.unsqueeze(0).repeat(B, 1, 1)  # (B, H*W, hidden_dim)
        
        return pos_emb
```

**类型2：Sine Position Embedding**
```python
class SinePositionEmbedding(nn.Module):
    def __init__(self, feature_size, hidden_dim):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_dim = hidden_dim
    
    def forward(self, B):
        H, W = self.feature_size
        y_embed = torch.arange(H).unsqueeze(1).repeat(1, W)  # (H, W)
        x_embed = torch.arange(W).unsqueeze(0).repeat(H, 1)  # (H, W)
        
        # 归一化到[0, 2π]
        y_embed = y_embed * (2 * math.pi / H)
        x_embed = x_embed * (2 * math.pi / W)
        
        # 生成sin/cos编码
        dim_t = torch.arange(self.hidden_dim // 2)
        dim_t = 10000 ** (2 * dim_t / self.hidden_dim)
        
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        
        pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=3).flatten(2)
        pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=3).flatten(2)
        
        pos_emb = torch.cat([pos_y, pos_x], dim=2)  # (H, W, hidden_dim)
        pos_emb = rearrange(pos_emb, 'h w c -> (h w) c')
        pos_emb = pos_emb.unsqueeze(0).repeat(B, 1, 1)
        
        return pos_emb
```

**配置**：
```yaml
net:
  - name: reconstruction
    kwargs:
      pos_embed_type: learned  # 或 sine
```

## 4. 模型参数量

### 4.1 参数统计

以MVTec-AD配置为例：

```python
Backbone (EfficientNet-B4, frozen): ~19M (不参与训练)
Neck (MFCN): ~0.14M
Reconstruction (UniAD):
  - input_proj: 272×256 = 0.07M
  - pos_embed: 196×256 = 0.05M
  - transformer: ~8M
  - output_proj: 256×272 = 0.07M
  
总可训练参数: ~8.3M
```

### 4.2 查看参数量

```python
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total: {total/1e6:.2f}M')
    print(f'Trainable: {trainable/1e6:.2f}M')
```

## 5. 模型优化技巧

### 5.1 梯度裁剪

```python
# tools/train_val.py
clip_max_norm = config.trainer.clip_max_norm  # 0.1
torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
```

**作用**：防止梯度爆炸

### 5.2 混合精度训练

```python
# 可选优化
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**作用**：加速训练，减少显存占用

### 5.3 模型并行

```python
# tools/train_val.py
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model, device_ids=[local_rank])
```

**作用**：多GPU训练加速

## 6. 调试技巧

### 6.1 打印模型结构

```python
from torchsummary import summary

model = ModelHelper(config).cuda()
summary(model, input_size=(3, 224, 224))
```

### 6.2 可视化特征图

```python
# 在forward中保存中间特征
def forward(self, x):
    feat = self.backbone(x)
    # 保存特征图
    torch.save(feat[0], 'debug_feat.pth')
    ...
```

### 6.3 检查梯度流

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f'{name}: grad_norm={param.grad.norm().item():.6f}')
```

## 7. 常见问题

### Q1: 为什么要冻结Backbone？
A: 预训练特征已经足够好，冻结可以防止过拟合，加快训练速度。

### Q2: Feature Jitter的scale如何选择？
A: 默认20.0适用于大多数情况。如果模型过拟合，可以增大；如果欠拟合，可以减小。

### Q3: Neighbor Mask的大小如何设置？
A: 默认[7,7]在14×14特征图上表现良好。更大的邻域允许更全局的注意力，更小的邻域更局部。

### Q4: 如何选择位置编码类型？
A: learned和sine都可以。learned可学习但参数多，sine固定但泛化好。默认learned效果略好。

## 8. 总结

UniAD模型的关键设计：

1. **预训练Backbone**：提供强大的特征提取能力
2. **MFCN融合**：整合多尺度信息
3. **Transformer重建**：灵活建模特征关系
4. **Feature Jitter**：训练时增强鲁棒性
5. **Neighbor Mask**：保持局部性，减少计算

下一步：阅读 [数据处理](DATASETS.md) 了解数据集加载和预处理流程。
