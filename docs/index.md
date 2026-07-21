# 算法工程师的 Infra 入门

> 一本"边学边记"的书，从一名算法工程师的视角，系统补齐 infra 知识。
> 力求**正确**：书中的结论尽量在真机上跑出数据来验证，具体数字只当示例，讲的是规律。

## 怎么读这本书

- **正文**讲概念，配**真机实验**（代码在每章附录里，可直接复制运行）。
- **术语**首次出现处有脚注，全称与解释汇总在[术语表](glossary.md)。
- 每章末尾有"补充"小节，展开硬件/工程背景。

## 全书大纲

完整课程见[**学习路线图**](roadmap.md)。总目标：从算法工程师视角吃透推理与训练 infra，
最终能**优化框架提速**、**优化算子 (kernel)**，并能**高效训练大模型**。分七部分：

| 部分 | 主题 | 你会获得 |
|------|------|----------|
| 一 | 推理与服务 | 看懂并配置高性能推理服务（🔜 进行中） |
| 二 | GPU 架构与算子优化 ★ | 手写/优化算子（reduction/gemm/scan…），用 profiler 定位瓶颈 |
| 三 | 框架与编译优化 ★ | 用 torch.compile / CUDA Graph 调优真实模型 |
| 四 | 分布式训练：并行与通信 | 设计并行策略（含 MoE），判断并调优通信瓶颈 |
| 五 | 大模型训练工程与规模化 ★ | 显存/数据/容错/稳定性/MFU/框架/集群全打通 |
| 六 | 综合实战 | 端到端"定位瓶颈 → 落地提速"（推理 + 训练） |
| 七 | 拓展与面试 | C++/OS/网络/ML-CV 速览 + AI infra 面试专题 |

★ = 核心目标（算子优化、框架提速、大模型训练工程）。

**第一部分当前进度**：

- ✅ 第 1 章 [推理的两个阶段：Prefill 与 Decode](chapters/01-inference/01-prefill-decode.md)
  （含[补充 A · HBM](chapters/01-inference/A1-gpu-storage-hbm.md)、[补充 B · CUDA 同步](chapters/01-inference/A2-cuda-async-sync-benchmark.md)）
- ✅ 第 2 章 [KV Cache：原理与显存计算](chapters/01-inference/02-kv-cache.md)
- ✅ 第 3 章 [Batching：static → continuous batching](chapters/01-inference/03-batching.md)
- ✅ 第 4 章 [PagedAttention 与显存管理](chapters/01-inference/04-paged-attention.md)
- ✅ 第 5 章 [前缀复用（Prefix Caching / RadixAttention）](chapters/01-inference/05-prefix-caching.md)
- ✅ 第 6 章 [量化：省显存与提速](chapters/01-inference/06-quantization.md)
- ✅ 第 7 章 [投机解码（Speculative Decoding）](chapters/01-inference/07-speculative-decoding.md)
- ✅ 第 8 章 [服务指标与压测（TTFT/TPOT/吞吐/goodput）](chapters/01-inference/08-serving-metrics.md)
- 🔜 第 9 章 推理引擎全景（vLLM / SGLang / TensorRT-LLM）

## 学习环境

- 软件：torch 2.8 (CUDA) · transformers 4.55 · numpy 1.26
- 实验用小模型：Qwen3-0.6B（够快、够说明问题）
