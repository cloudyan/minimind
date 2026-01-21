# 第一章：项目结构概览

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
