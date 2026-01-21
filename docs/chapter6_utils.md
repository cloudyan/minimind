# 第六章：核心训练工具函数

## 6.1 学习率调度
```python
def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))
```

## 6.2 检查点管理
```python
def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    # 支持断点续训和跨GPU数量恢复
    if model is not None:  # 保存模式
        # 保存模型、优化器、训练状态
    else:  # 加载模式
        # 加载检查点数据
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            # 自动处理GPU数量变化对step的影响
            return ckp_data
    return None
```
