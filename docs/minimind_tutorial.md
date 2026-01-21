# MiniMind 源码学习教程

## 第一章：项目结构概览

MiniMind是一个从零开始构建的超小语言模型项目，具有完整的训练、推理和部署功能。项目整体结构如下：

```
/Volumes/data/code/github.com/cloudyan/minimind/
├── README.md                    # 项目介绍和使用说明
├── requirements.txt             # 依赖库
├── dataset/                     # 数据集相关
│   ├── lm_dataset.py            # 数据集处理类
│   └── ...
├── model/                       # 模型定义
│   ├── model_minimind.py        # 核心模型架构
│   ├── model_lora.py            # LoRA微调实现
│   └── ...
├── trainer/                     # 训练脚本
│   ├── train_pretrain.py        # 预训练
│   ├── train_full_sft.py        # 全参数微调
│   ├── train_dpo.py             # DPO训练
│   ├── train_lora.py            # LoRA微调
│   └── trainer_utils.py         # 训练工具函数
├── eval_llm.py                  # 模型推理评估
└── scripts/                     # 辅助脚本
    └── train_tokenizer.py       # 分词器训练
```

## 第二章：模型架构详解

### 2.1 模型配置类 MiniMindConfig

MiniMind使用`MiniMindConfig`类定义模型参数，包含基础配置和MoE配置：

```python
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            hidden_size: int = 512,      # 隐藏层维度 (26M模型为512，104M模型为768)
            num_hidden_layers: int = 8,  # 隐藏层数量
            num_attention_heads: int = 8, # 注意力头数
            num_key_value_heads: int = 2, # KV头数
            vocab_size: int = 6400,      # 词表大小
            rope_theta: int = 1000000.0, # RoPE的基值
            use_moe: bool = False,       # 是否使用MoE
            # ... MoE相关配置
    ):
        # 配置初始化
```

### 2.2 核心组件实现

#### RMSNorm归一化层
```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
```

#### RoPE位置编码
```python
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    # 预计算旋转位置编码
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # ... RoPE外推实现
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # 应用旋转位置编码
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed
```

#### 注意力机制
```python
class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        # 设置多头注意力参数
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        
        # 定义线性投影层
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # 注意力计算流程
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])
        
        # KV Cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # GQA (Grouped Query Attention)
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        # Flash Attention或普通注意力计算
        if self.flash and seq_len > 1:
            # 使用Flash Attention
        else:
            # 使用普通注意力计算
```

#### 前馈网络 (FFN) 与 MoE
```python
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

MoE (Mixture of Experts) 实现：
```python
class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        # 门控机制实现

class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
```

## 第三章：数据处理流程

### 3.1 数据集类
MiniMind定义了多种类型的数据集处理类：

- `PretrainDataset`: 预训练数据集
- `SFTDataset`: 监督微调数据集
- `DPODataset`: DPO训练数据集
- `RLAIFDataset`: 强化学习数据集

以SFT数据集为例：
```python
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def _create_chat_prompt(self, cs):
        # 构建聊天模板提示
        messages = cs.copy()
        tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def _generate_loss_mask(self, input_ids):
        # 生成损失掩码，只计算assistant回复部分的损失
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        loss_mask = self._generate_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask
```

## 第四章：训练流程详解

### 4.1 预训练 (Pretrain)
预训练脚本 `train_pretrain.py` 实现了语言模型的自回归预训练：

```python
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        
        # 动态学习率调度
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(X)
            # 计算带掩码的损失
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss  # 添加MoE辅助损失
            loss = loss / args.accumulation_steps

        # 梯度累积与反向传播
        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
```

### 4.2 DPO训练
DPO (Direct Preference Optimization) 是一种强化学习方法，无需奖励模型：

```python
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # 计算参考模型和策略模型的对数概率
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # 分离chosen和rejected样本
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```

### 4.3 LoRA微调
LoRA (Low-Rank Adaptation) 是参数高效微调方法：

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        self.A.weight.data.normal_(mean=0.0, std=0.02)  # 高斯初始化
        self.B.weight.data.zero_()  # 零初始化

    def forward(self, x):
        return self.B(self.A(x))

def apply_lora(model, rank=8):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora
```

## 第五章：推理与评估

`eval_llm.py` 提供了完整的模型推理接口：

```python
def main():
    # ... 参数解析 ...
    
    conversation = []
    model, tokenizer = init_model(args)

    for prompt in prompts:
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        # 应用聊天模板
        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        if args.weight == 'reason': templates["enable_thinking"] = True
        inputs = tokenizer.apply_chat_template(**templates) if args.weight != 'pretrain' else (tokenizer.bos_token + prompt)
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        # 生成回复
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1.0
        )
```

## 第六章：核心训练工具函数

### 6.1 学习率调度
```python
def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))
```

### 6.2 检查点管理
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

## 总结

MiniMind项目是一个完整的、从零开始构建的超小语言模型实现，包含以下核心特性：

1. **Transformer架构**：使用标准的Decoder-Only结构，包含RoPE、RMSNorm、GQA等现代LLM技术
2. **MoE支持**：实现了Mixture of Experts架构，支持路由专家和共享专家
3. **完整的训练流程**：预训练、SFT、DPO、PPO/GRPO/SPO、LoRA等
4. **高效实现**：使用Flash Attention、梯度累积、混合精度等优化技术
5. **实用功能**：支持RoPE长度外推、KV Cache、断点续训等

该项目代码结构清晰，功能完整，是学习和理解现代大语言模型架构和训练流程的良好范例。
