# MiniMind 源码学习教程

## 目录

- [前言：大模型概念入门](./chapter0_concept.md)
- [第一章：项目结构概览](./chapter1_structure.md)
- [第二章：模型架构详解](./chapter2_model.md)
- [第三章：数据处理流程](./chapter3_data.md)
- [第四章：训练流程详解](./chapter4_training.md)
- [第五章：推理与评估](./chapter5_inference.md)
- [第六章：核心训练工具函数](./chapter6_utils.md)

## 总结

MiniMind项目是一个完整的、从零开始构建的超小语言模型实现，包含以下核心特性：

1. **Transformer架构**：使用标准的Decoder-Only结构，包含RoPE、RMSNorm、GQA等现代LLM技术
2. **MoE支持**：实现了Mixture of Experts架构，支持路由专家和共享专家
3. **完整的训练流程**：预训练、SFT、DPO、PPO/GRPO/SPO、LoRA等
4. **高效实现**：使用Flash Attention、梯度累积、混合精度等优化技术
5. **实用功能**：支持RoPE长度外推、KV Cache、断点续训等

该项目代码结构清晰，功能完整，是学习和理解现代大语言模型架构和训练流程的良好范例。
