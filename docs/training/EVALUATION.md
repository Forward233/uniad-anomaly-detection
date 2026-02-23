# UniAD 评估和可视化

本文档介绍 UniAD 的评估指标计算、结果分析和可视化工具。

## 1. 评估流程

### 1.1 评估脚本

```bash
cd experiments/MVTec-AD
bash eval_torch.sh 1 0
```

### 1.2 评估流程

```
加载测试数据 → 模型推理 → 计算异常分数 → 计算评估指标 → 可视化结果
```

## 2. 异常分数计算

### 2.1 像素级异常分数

**实现**（`utils/eval_helper.py`）：
```python
def compute_anomaly_score(feature, reconstructed):
    # 1. 计算重建误差
    error = (feature - reconstructed) ** 2  # (B, C, H, W)
    
    # 2. 在通道维度求和
    score = error.sum(dim=1, keepdim=True)  # (B, 1, H, W)
    
    # 3. 上采样到原图尺寸
    score = F.interpolate(score, size=(224, 224), mode='bilinear')
    
    # 4. 归一化到[0, 1]
    score = (score - score.min()) / (score.max() - score.min() + 1e-8)
    
    return score
```

### 2.2 图像级异常分数

```python
def compute_image_score(pixel_scores):
    # 方法1: 最大值
    image_score = pixel_scores.max()
    
    # 方法2: 平均值
    # image_score = pixel_scores.mean()
    
    # 方法3: Top-K平均
    # top_k = pixel_scores.flatten().topk(k=100).values
    # image_score = top_k.mean()
    
    return image_score
```

## 3. 评估指标

### 3.1 Pixel AUC

**含义**：像素级异常检测的AUC

**计算**：
```python
from sklearn.metrics import roc_auc_score

def compute_pixel_auc(pred_masks, gt_masks):
    # pred_masks: (N, H, W) 预测的异常分数
    # gt_masks: (N, H, W) ground truth掩码
    
    # 展平
    pred = pred_masks.flatten()
    gt = gt_masks.flatten()
    
    # 计算AUC
    auc = roc_auc_score(gt, pred)
    
    return auc
```

### 3.2 Image AUC

**含义**：图像级异常检测的AUC

**计算**：
```python
def compute_image_auc(image_scores, labels):
    # image_scores: (N,) 每张图像的异常分数
    # labels: (N,) 图像标签 (0=正常, 1=异常)
    
    auc = roc_auc_score(labels, image_scores)
    return auc
```

### 3.3 Mean/Std/Max AUC

**配置**：
```yaml
evaluator:
  metrics:
    auc:
      - name: mean     # 平均值作为图像分数
      - name: std      # 标准差作为图像分数
      - name: max      # 最大值作为图像分数
        kwargs:
          avgpool_size: [16, 16]
      - name: pixel    # 像素级AUC
```

**实现**：
```python
def compute_auc_variants(pred_masks, labels, gt_masks):
    results = {}
    
    # Pixel AUC
    results['pixel_auc'] = compute_pixel_auc(pred_masks, gt_masks)
    
    # Mean AUC
    mean_scores = pred_masks.mean(dim=(1, 2))
    results['mean_auc'] = roc_auc_score(labels, mean_scores)
    
    # Std AUC
    std_scores = pred_masks.std(dim=(1, 2))
    results['std_auc'] = roc_auc_score(labels, std_scores)
    
    # Max AUC
    max_scores = pred_masks.max(dim=1)[0].max(dim=1)[0]
    results['max_auc'] = roc_auc_score(labels, max_scores)
    
    return results
```

### 3.4 Per-Class AUC

```python
def compute_per_class_auc(pred_masks, labels, clsnames):
    class_aucs = {}
    
    for cls in set(clsnames):
        # 筛选该类别的样本
        indices = [i for i, c in enumerate(clsnames) if c == cls]
        
        # 计算该类别的AUC
        cls_pred = pred_masks[indices]
        cls_label = labels[indices]
        
        auc = compute_pixel_auc(cls_pred, cls_label)
        class_aucs[cls] = auc
    
    return class_aucs
```

## 4. 评估代码示例

### 4.1 完整评估函数

```python
def evaluate(model, test_loader, config):
    model.eval()
    
    all_pred_masks = []
    all_gt_masks = []
    all_labels = []
    all_clsnames = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].cuda()
            labels = batch['label']
            gt_masks = batch.get('mask', None)
            clsnames = batch['clsname']
            
            # 前向传播
            outputs = model(images)
            
            # 计算异常分数
            pred_masks = compute_anomaly_score(
                outputs['feature'], 
                outputs['reconstructed']
            )
            
            # 保存结果
            all_pred_masks.append(pred_masks.cpu())
            all_labels.extend(labels.numpy())
            all_clsnames.extend(clsnames)
            if gt_masks is not None:
                all_gt_masks.append(gt_masks)
    
    # 合并结果
    all_pred_masks = torch.cat(all_pred_masks, dim=0)
    all_gt_masks = torch.cat(all_gt_masks, dim=0) if all_gt_masks else None
    all_labels = np.array(all_labels)
    
    # 计算指标
    metrics = {}
    
    # Pixel AUC
    if all_gt_masks is not None:
        metrics['pixel_auc'] = compute_pixel_auc(all_pred_masks, all_gt_masks)
    
    # Image AUC
    image_scores = all_pred_masks.max(dim=1)[0].max(dim=1)[0]
    metrics['image_auc'] = roc_auc_score(all_labels, image_scores)
    
    # Per-Class AUC
    class_aucs = compute_per_class_auc(all_pred_masks, all_labels, all_clsnames)
    metrics.update({f'auc_{cls}': auc for cls, auc in class_aucs.items()})
    
    return metrics
```

## 5. 可视化

### 5.1 异常热图

**文件位置**：`utils/vis_helper.py`

```python
def visualize_heatmap(image, pred_mask, gt_mask=None, save_path=None):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3 if gt_mask is not None else 2, figsize=(12, 4))
    
    # 原图
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 预测热图
    axes[1].imshow(image)
    axes[1].imshow(pred_mask, alpha=0.5, cmap='jet')
    axes[1].set_title('Predicted Anomaly')
    axes[1].axis('off')
    
    # Ground Truth
    if gt_mask is not None:
        axes[2].imshow(image)
        axes[2].imshow(gt_mask, alpha=0.5, cmap='gray')
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    else:
        plt.show()
    
    plt.close()
```

### 5.2 ROC曲线

```python
from sklearn.metrics import roc_curve

def plot_roc_curve(labels, scores, save_path=None):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    else:
        plt.show()
    
    plt.close()
```

### 5.3 批量可视化

```python
def visualize_batch_results(model, test_loader, save_dir, num_samples=10):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    count = 0
    with torch.no_grad():
        for batch in test_loader:
            if count >= num_samples:
                break
            
            images = batch['image'].cuda()
            gt_masks = batch.get('mask', None)
            filenames = batch['filename']
            
            outputs = model(images)
            pred_masks = compute_anomaly_score(
                outputs['feature'], 
                outputs['reconstructed']
            )
            
            for i in range(len(images)):
                if count >= num_samples:
                    break
                
                # 反归一化图像
                image = denormalize(images[i].cpu())
                pred = pred_masks[i, 0].cpu().numpy()
                gt = gt_masks[i, 0].numpy() if gt_masks is not None else None
                
                save_path = os.path.join(save_dir, f'result_{count}.png')
                visualize_heatmap(image, pred, gt, save_path)
                
                count += 1
```

## 6. 结果分析

### 6.1 混淆矩阵

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(labels, predictions, threshold=0.5):
    pred_binary = (predictions > threshold).astype(int)
    cm = confusion_matrix(labels, pred_binary)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                   display_labels=['Normal', 'Anomaly'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()
```

### 6.2 分类报告

```python
from sklearn.metrics import classification_report

def print_classification_report(labels, predictions, threshold=0.5):
    pred_binary = (predictions > threshold).astype(int)
    report = classification_report(labels, pred_binary, 
                                   target_names=['Normal', 'Anomaly'])
    print(report)
```

### 6.3 阈值选择

```python
def find_best_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # 方法1: Youden's J statistic
    j_scores = tpr - fpr
    best_idx = j_scores.argmax()
    best_threshold = thresholds[best_idx]
    
    # 方法2: 最小化距离
    # distances = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
    # best_idx = distances.argmin()
    # best_threshold = thresholds[best_idx]
    
    return best_threshold
```

## 7. 实验结果示例

### 7.1 MVTec-AD结果

| 类别 | Pixel AUC | Image AUC |
|------|-----------|-----------|
| bottle | 98.5 | 99.2 |
| cable | 97.3 | 98.1 |
| capsule | 98.9 | 99.5 |
| ... | ... | ... |
| **Mean** | **98.1** | **98.7** |

### 7.2 CIFAR-10结果

| 配置 | Image AUC |
|------|-----------|
| 01234 | 85.3 |
| 02468 | 87.1 |
| 13579 | 84.2 |
| 56789 | 86.5 |

## 8. 可视化工具

### 8.1 重建结果可视化

**脚本**：`tools/vis_recon.py`

```bash
python tools/vis_recon.py --config experiments/MVTec-AD/config.yaml
```

### 8.2 Query向量可视化

**脚本**：`tools/vis_query.py`

```bash
python tools/vis_query.py --config experiments/MVTec-AD/config.yaml
```

### 8.3 特征图可视化

```python
def visualize_features(feature_maps, save_path=None):
    # feature_maps: (C, H, W)
    C, H, W = feature_maps.shape
    
    # 选择前16个通道
    num_show = min(16, C)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    for i, ax in enumerate(axes.flat):
        if i < num_show:
            feat = feature_maps[i].cpu().numpy()
            ax.imshow(feat, cmap='viridis')
            ax.set_title(f'Channel {i}')
        ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
```

## 9. 调试技巧

### 9.1 检查异常分数分布

```python
def analyze_score_distribution(scores, labels):
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    plt.figure(figsize=(10, 5))
    plt.hist(normal_scores, bins=50, alpha=0.5, label='Normal', density=True)
    plt.hist(anomaly_scores, bins=50, alpha=0.5, label='Anomaly', density=True)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Score Distribution')
    plt.show()
    
    print(f'Normal: mean={normal_scores.mean():.3f}, std={normal_scores.std():.3f}')
    print(f'Anomaly: mean={anomaly_scores.mean():.3f}, std={anomaly_scores.std():.3f}')
```

### 9.2 可视化失败案例

```python
def visualize_failure_cases(model, test_loader, num_cases=10):
    # 找出预测错误的样本
    failures = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].cuda()
            labels = batch['label'].numpy()
            
            outputs = model(images)
            scores = compute_image_scores(outputs)
            
            # 预测错误的样本
            threshold = 0.5
            pred_labels = (scores > threshold).cpu().numpy()
            wrong_idx = np.where(pred_labels != labels)[0]
            
            for idx in wrong_idx:
                failures.append({
                    'image': images[idx].cpu(),
                    'true_label': labels[idx],
                    'pred_score': scores[idx].item()
                })
                
                if len(failures) >= num_cases:
                    break
            
            if len(failures) >= num_cases:
                break
    
    # 可视化
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, (ax, failure) in enumerate(zip(axes.flat, failures)):
        image = denormalize(failure['image'])
        ax.imshow(image)
        ax.set_title(f"True: {failure['true_label']}, Score: {failure['pred_score']:.3f}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()
```

## 10. 性能优化

### 10.1 批量推理

```python
@torch.no_grad()
def batch_inference(model, images):
    # 使用更大的batch size加速推理
    model.eval()
    outputs = model(images)
    return outputs
```

### 10.2 半精度推理

```python
@torch.no_grad()
@torch.cuda.amp.autocast()
def inference_fp16(model, images):
    outputs = model(images)
    return outputs
```

## 11. 导出结果

### 11.1 保存评估结果

```python
import json

def save_evaluation_results(metrics, save_path):
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Results saved to {save_path}')
```

### 11.2 生成报告

```python
def generate_report(metrics, save_path):
    with open(save_path, 'w') as f:
        f.write('# Evaluation Report\n\n')
        f.write('## Overall Metrics\n\n')
        f.write(f"- Pixel AUC: {metrics['pixel_auc']:.4f}\n")
        f.write(f"- Image AUC: {metrics['image_auc']:.4f}\n\n")
        
        f.write('## Per-Class Results\n\n')
        for key, value in metrics.items():
            if key.startswith('auc_'):
                cls = key.replace('auc_', '')
                f.write(f"- {cls}: {value:.4f}\n")
```

## 12. 常见问题

### Q1: AUC很低怎么办？
A: 检查模型是否正确训练、特征是否正常、阈值是否合适。

### Q2: 如何提高像素级检测精度？
A: 使用更精细的特征、调整上采样方法、后处理平滑。

### Q3: 可视化结果保存在哪里？
A: 配置文件中`evaluator.vis_compound.save_dir`指定的目录。

### Q4: 如何评估自定义数据集？
A: 准备ground truth掩码，使用相同的评估流程。

## 13. 总结

UniAD评估的关键点：

1. **异常分数**：重建误差作为异常指标
2. **多层次评估**：像素级和图像级AUC
3. **可视化工具**：热图、ROC曲线、特征图
4. **Per-Class分析**：每个类别单独评估
5. **失败案例分析**：找出预测错误的样本

下一步：阅读 [进阶实践](../advanced/ADVANCED.md) 了解如何自定义开发和优化。
