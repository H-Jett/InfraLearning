# 算法工程师的 Infra 入门

> 一本"边学边记"的书，从一名算法工程师的视角，系统补齐 infra 知识。
> 力求**正确**：书中的结论尽量在真机上跑出数据来验证，具体数字只当示例，讲的是规律。

## 怎么读这本书

- **正文**讲概念，配**真机实验**（代码在每章附录里，可直接复制运行）。
- **术语**首次出现处有脚注，全称与解释汇总在[术语表](glossary.md)。
- 每章末尾有"补充"小节，展开硬件/工程背景。

## 全书大纲

### 第一部分 · 推理与服务（进行中）

| 章 | 主题 | 状态 |
|----|------|------|
| 1 | [推理的两个阶段：Prefill 与 Decode](chapters/01-inference/01-prefill-decode.md) | ✅ 已完成 |
| — | [补充 A：GPU 存储层级与 HBM](chapters/01-inference/A1-gpu-storage-hbm.md) | ✅ |
| — | [补充 B：CUDA 异步/同步与基准测试](chapters/01-inference/A2-cuda-async-sync-benchmark.md) | ✅ |
| 2 | KV Cache：原理与显存计算 | 待学 |
| 3 | Batching：从 static 到 continuous batching | 待学 |
| 4 | PagedAttention 与显存管理 | 待学 |
| 5 | 前缀复用（Prefix Caching / RadixAttention） | 待学 |
| 6 | 量化：省显存与提速 | 待学 |
| 7 | 投机解码（Speculative Decoding） | 待学 |
| 8 | 关键指标：TTFT / TPOT / 吞吐 / Goodput | 待学 |
| 9 | 引擎实战：vLLM / SGLang | 待学 |

### 第二部分 · 分布式训练（待开始）

- 数据并行 / 张量并行 / 流水线并行
- ZeRO 与 FSDP
- NCCL 与集合通信
- 3D 并行与实战

## 学习环境

- 软件：torch 2.8 (CUDA) · transformers 4.55 · numpy 1.26
- 实验用小模型：Qwen3-0.6B（够快、够说明问题）
