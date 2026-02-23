# UniAD 训练流程

本文档详细介绍 UniAD 的训练主流程、配置文件、损失函数和优化器。

## 1. 训练脚本概览

### 1.1 主训练文件

**文件位置**：`tools/train_val.py`

**执行流程**：
```
解析配置 → 初始化分布式 → 构建模型 → 构建优化器 → 训练循环 → 验证评估 → 保存模型
```

### 1.2 启动命令

```bash
cd experiments/MVTec-AD
bash train_torch.sh 1 0
```

**train_torch.sh内容**：
```bash
export PYTHONPATH=../../:$PYTHONPATH
CUDA_VISIBLE_DEVICES=$2
python -m torch.distributed.launch --nproc_per_node=$1 ../../tools/train_val.py
```

**参数说明**：
- `$1`: GPU数量（nproc_per_node）
- `$2`: CUDA设备ID

## 2. 配置文件

### 2.1 配置文件结构

**文件位置**：`experiments/MVTec-AD/config.yaml`

```yaml
version: v1.0.0
random_seed: 133
port: 11111

dataset:
  type: custom
  batch_size: 8
  workers: 4
  input_size: [224, 224]
  pixel_mean: [0.485, 0.456, 0.406]
  pixel_std: [0.229, 0.224, 0.225]
  train:
    meta_file: ../../data/MVTec-AD/train.json
  test:
    meta_file: ../../data/MVTec-AD/test.json

criterion:
  - name: FeatureMSELoss
    type: FeatureMSELoss
    kwargs:
      weight: 1.0

trainer:
  max_epoch: 1000
  clip_max_norm: 0.1
  val_freq_epoch: 10
  print_freq_step: 1
  tb_freq_step: 1
  optimizer:
    type: AdamW
    kwargs:
      lr: 0.0001
      betas: [0.9, 0.999]
      weight_decay: 0.0001
  lr_scheduler:
    type: StepLR
    kwargs:
      step_size: 800
      gamma: 0.1

saver:
  auto_resume: False
  always_save: False
  save_dir: checkpoints/
  log_dir: log/

net:
  - name: backbone
    type: models.backbones.efficientnet_b4
    frozen: True
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

### 2.2 关键配置说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| batch_size | 批次大小 | 8 |
| max_epoch | 最大训练轮数 | 1000 |
| lr | 学习率 | 0.0001 |
| clip_max_norm | 梯度裁剪阈值 | 0.1 |
| val_freq_epoch | 验证频率 | 10 |
| step_size | 学习率衰减步长 | 800 |
| gamma | 学习率衰减系数 | 0.1 |

## 3. 训练主流程

### 3.1 main函数

```python
def main():
    # 1. 解析参数和配置
    args = parser.parse_args()
    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    
    # 2. 初始化分布式
    rank, world_size = setup_distributed(port=config.port)
    config = update_config(config)
    
    # 3. 设置随机种子
    set_random_seed(config.random_seed)
    
    # 4. 创建logger
    logger = create_logger(config.log_path)
    
    # 5. 构建数据加载器
    train_loader = build_dataloader(config, training=True)
    test_loader = build_dataloader(config, training=False)
    
    # 6. 构建模型
    model = ModelHelper(config)
    model = model.cuda()
    model = DDP(model, device_ids=[rank])
    
    # 7. 构建损失函数
    criterion = build_criterion(config.criterion)
    
    # 8. 构建优化器和学习率调度器
    optimizer = get_optimizer(config.trainer.optimizer, model)
    lr_scheduler = get_scheduler(config.trainer.lr_scheduler, optimizer)
    
    # 9. 自动恢复（如果配置）
    start_epoch = 0
    if config.saver.auto_resume:
        start_epoch = load_state(checkpoint_path, model, optimizer)
    
    # 10. 训练循环
    for epoch in range(start_epoch, config.trainer.max_epoch):
        # 训练一个epoch
        train_one_epoch(epoch, model, train_loader, optimizer, 
                       lr_scheduler, criterion, config, logger)
        
        # 验证
        if (epoch + 1) % config.trainer.val_freq_epoch == 0:
            metrics = validate(epoch, model, test_loader, config, logger)
            
            # 保存最优模型
            is_best = metrics[config.evaluator.key_metric] > best_metric
            if is_best:
                best_metric = metrics[config.evaluator.key_metric]
                save_checkpoint(state, is_best, config.save_path)
```

### 3.2 train_one_epoch函数

```python
def train_one_epoch(epoch, model, train_loader, optimizer, 
                    lr_scheduler, criterion, config, logger):
    model.train()
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    end = time.time()
    for i, batch in enumerate(train_loader):
        # 1. 数据加载时间
        data_time.update(time.time() - end)
        
        # 2. 数据移到GPU
        images = batch['image'].cuda()
        
        # 3. 前向传播
        outputs = model(images)
        
        # 4. 计算损失
        loss_dict = criterion(outputs)
        loss = sum(loss_dict.values())
        
        # 5. 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 6. 梯度裁剪
        if config.trainer.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config.trainer.clip_max_norm
            )
        
        # 7. 更新参数
        optimizer.step()
        
        # 8. 更新学习率
        lr_scheduler.step()
        
        # 9. 记录统计
        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        # 10. 打印日志
        if i % config.trainer.print_freq_step == 0:
            logger.info(
                f'Epoch: [{epoch}/{config.trainer.max_epoch}]\t'
                f'Iter: [{i}/{len(train_loader)}]\t'
                f'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                f'Data {data_time.val:.2f} ({data_time.avg:.2f})\t'
                f'Loss {losses.val:.5f} ({losses.avg:.5f})\t'
                f'LR {optimizer.param_groups[0]["lr"]:.5f}'
            )
        
        # 11. TensorBoard记录
        if i % config.trainer.tb_freq_step == 0:
            writer.add_scalar('train/loss', losses.val, global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
```

### 3.3 validate函数

```python
def validate(epoch, model, test_loader, config, logger):
    model.eval()
    
    results = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].cuda()
            labels = batch['label']
            masks = batch.get('mask', None)
            
            # 前向传播
            outputs = model(images)
            
            # 保存结果
            results.append({
                'pred': outputs['pred'].cpu(),
                'label': labels,
                'mask': masks,
                'clsname': batch['clsname']
            })
    
    # 合并所有结果
    results = merge_together(results)
    
    # 计算评估指标
    metrics = performances(results, config.evaluator.metrics)
    
    # 打印日志
    log_metrics(logger, metrics, epoch)
    
    return metrics
```

## 4. 损失函数

### 4.1 FeatureMSELoss

**文件位置**：`utils/criterion_helper.py`

**实现**：
```python
class FeatureMSELoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.mse = nn.MSELoss()
    
    def forward(self, outputs):
        # outputs包含原始特征和重建特征
        feature = outputs['feature']  # (B, C, H, W)
        reconstructed = outputs['reconstructed']  # (B, C, H, W)
        
        # 计算MSE损失
        loss = self.mse(reconstructed, feature)
        
        return {'FeatureMSELoss': loss * self.weight}
```

**损失计算流程**：
```
1. Backbone提取特征: feature
2. Neck融合特征: fused_feature
3. Reconstruction重建: reconstructed_feature
4. 计算MSE: ||reconstructed_feature - fused_feature||^2
5. 反向传播更新参数
```

### 4.2 多损失组合

**配置示例**：
```yaml
criterion:
  - name: FeatureMSELoss
    type: FeatureMSELoss
    kwargs:
      weight: 1.0
  - name: L1Loss
    type: L1Loss
    kwargs:
      weight: 0.5
```

**实现**：
```python
def build_criterion(criterion_cfg):
    criterions = []
    for cfg in criterion_cfg:
        criterion_cls = eval(cfg['type'])
        criterion = criterion_cls(**cfg.get('kwargs', {}))
        criterions.append(criterion)
    
    def combined_criterion(outputs):
        loss_dict = {}
        for criterion in criterions:
            loss_dict.update(criterion(outputs))
        return loss_dict
    
    return combined_criterion
```

## 5. 优化器

### 5.1 AdamW优化器

**配置**：
```yaml
trainer:
  optimizer:
    type: AdamW
    kwargs:
      lr: 0.0001
      betas: [0.9, 0.999]
      weight_decay: 0.0001
```

**构建**：
```python
def get_optimizer(optimizer_cfg, model):
    optimizer_type = optimizer_cfg['type']
    optimizer_kwargs = optimizer_cfg.get('kwargs', {})
    
    # 只优化requires_grad=True的参数
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(params, **optimizer_kwargs)
    elif optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(params, **optimizer_kwargs)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_type}')
    
    return optimizer
```

### 5.2 参数分组（进阶）

```python
def get_optimizer_with_groups(config, model):
    # 不同层使用不同学习率
    backbone_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config.lr * 0.1},
        {'params': other_params, 'lr': config.lr}
    ], betas=config.betas, weight_decay=config.weight_decay)
    
    return optimizer
```

## 6. 学习率调度

### 6.1 StepLR

**配置**：
```yaml
trainer:
  lr_scheduler:
    type: StepLR
    kwargs:
      step_size: 800
      gamma: 0.1
```

**实现**：
```python
def get_scheduler(scheduler_cfg, optimizer):
    scheduler_type = scheduler_cfg['type']
    scheduler_kwargs = scheduler_cfg.get('kwargs', {})
    
    if scheduler_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, **scheduler_kwargs
        )
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **scheduler_kwargs
        )
    else:
        raise ValueError(f'Unknown scheduler: {scheduler_type}')
    
    return scheduler
```

**学习率变化**：
```
Epoch 0-799: lr = 0.0001
Epoch 800-1599: lr = 0.0001 * 0.1 = 0.00001
Epoch 1600+: lr = 0.00001 * 0.1 = 0.000001
```

### 6.2 其他调度器

**CosineAnnealing**：
```yaml
lr_scheduler:
  type: CosineAnnealingLR
  kwargs:
    T_max: 1000
    eta_min: 0.000001
```

**MultiStepLR**：
```yaml
lr_scheduler:
  type: MultiStepLR
  kwargs:
    milestones: [400, 800]
    gamma: 0.1
```

## 7. 梯度裁剪

### 7.1 为什么需要梯度裁剪？

防止梯度爆炸，稳定训练过程。

### 7.2 实现

```python
if config.trainer.clip_max_norm > 0:
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), 
        max_norm=config.trainer.clip_max_norm
    )
```

**配置**：
```yaml
trainer:
  clip_max_norm: 0.1
```

## 8. 模型保存和加载

### 8.1 保存策略

**只保存最优模型**：
```yaml
saver:
  always_save: False
  save_dir: checkpoints/
```

**每个epoch都保存**：
```yaml
saver:
  always_save: True
  save_dir: checkpoints/
```

### 8.2 Checkpoint结构

```python
checkpoint = {
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'lr_scheduler': lr_scheduler.state_dict(),
    'best_metric': best_metric,
    'config': config
}

torch.save(checkpoint, 'checkpoints/ckpt.pth.tar')
```

### 8.3 自动恢复训练

**配置**：
```yaml
saver:
  auto_resume: True
  load_path: checkpoints/ckpt.pth.tar
```

**实现**：
```python
if config.saver.auto_resume and os.path.exists(config.saver.load_path):
    checkpoint = torch.load(config.saver.load_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    start_epoch = checkpoint['epoch'] + 1
    best_metric = checkpoint['best_metric']
    logger.info(f'Resumed from epoch {start_epoch}')
```

## 9. 分布式训练

### 9.1 初始化

**文件位置**：`utils/dist_helper.py`

```python
def setup_distributed(port=None):
    if 'RANK' in os.environ:
        # 已由torch.distributed.launch设置
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # 单机单卡
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://localhost:{port}',
            rank=rank,
            world_size=world_size
        )
    
    return rank, world_size
```

### 9.2 DDP包装

```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = model.cuda()
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
```

### 9.3 DistributedSampler

```python
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(dataset, shuffle=True)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
```

### 9.4 多GPU训练命令

```bash
# 2个GPU
bash train_torch.sh 2 0,1

# 4个GPU
bash train_torch.sh 4 0,1,2,3
```

## 10. TensorBoard可视化

### 10.1 启动TensorBoard

```bash
tensorboard --logdir experiments/MVTec-AD/log
```

### 10.2 记录内容

```python
from tensorboardX import SummaryWriter

writer = SummaryWriter(config.log_dir)

# 记录损失
writer.add_scalar('train/loss', loss.item(), global_step)

# 记录学习率
writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

# 记录图像
writer.add_image('train/image', image, global_step)

# 记录模型图
writer.add_graph(model, images)
```

## 11. 训练技巧

### 11.1 Warm-up

```python
def get_lr_with_warmup(epoch, warmup_epochs=5, base_lr=0.0001):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        return base_lr
```

### 11.2 早停

```python
patience = 50
counter = 0
best_metric = 0

for epoch in range(max_epoch):
    metrics = validate(...)
    
    if metrics[key_metric] > best_metric:
        best_metric = metrics[key_metric]
        counter = 0
        save_checkpoint(...)
    else:
        counter += 1
    
    if counter >= patience:
        logger.info('Early stopping')
        break
```

### 11.3 学习率查找

```python
from torch_lr_finder import LRFinder

lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
lr_finder.plot()
```

## 12. 常见问题

### Q1: 如何调整学习率？
A: 修改`config.yaml`中的`trainer.optimizer.kwargs.lr`，建议范围0.0001-0.001。

### Q2: 训练loss不下降怎么办？
A: 检查学习率是否过小、梯度是否正常、数据是否正确加载。

### Q3: 如何在训练中途修改配置？
A: 修改config.yaml后，使用auto_resume从checkpoint恢复训练。

### Q4: 显存不足怎么办？
A: 减小batch_size、input_size，或使用梯度累积。

### Q5: 如何查看训练进度？
A: 使用`tail -f log/train.log`或TensorBoard。

## 13. 总结

UniAD训练流程的关键点：

1. **配置驱动**：所有参数通过config.yaml配置
2. **分布式支持**：使用DDP实现多GPU训练
3. **MSE损失**：重建误差作为训练目标
4. **AdamW优化**：自适应学习率优化器
5. **学习率调度**：StepLR周期性降低学习率
6. **梯度裁剪**：防止梯度爆炸
7. **定期验证**：每10个epoch验证一次

下一步：阅读 [评估可视化](EVALUATION.md) 了解如何评估模型性能和可视化结果。
