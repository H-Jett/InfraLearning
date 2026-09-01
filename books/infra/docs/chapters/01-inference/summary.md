# 第一部分小结 · 推理与服务（速查）

> 一页回顾整个第一部分。看不懂某条就回对应章节。

## 一条主线

> **LLM 推理 = 一次"吃满算力"的 prefill（compute-bound） + 一长串"喂不饱算力"的 decode（memory-bound）。
> 几乎所有优化，都是在跟 decode 的 memory-bound 特性、以及它带来的 KV cache 显存压力作斗争。**

## 两个必记公式

- **TPOT 下限 ≈ 模型字节 ÷ 显存带宽**（第 1 章）——decode 是 bandwidth-bound 时的速度墙。
- **KV_bytes = 2 × 层数 × KV头数 × head_dim × 序列长 × batch × 每元素字节**（第 2 章）
  —— 两个最容易漏：① 乘层数；② 是 KV 头不是 attention 头。

## 九章速览

| 章 | 解决什么问题 | 关键结论 / 数字 | 何时用 |
|----|------------|----------------|--------|
| 1 Prefill/Decode | 推理为什么慢 | decode 1 token ≈ prefill 4096 token（memory-bound） | —（地基） |
| 2 KV Cache | 缓存了什么、多大 | 满上下文单序列 KV 可达模型权重的 3.9× | —（地基） |
| 3 Batching | 怎么提吞吐 | 权重搬一次喂一批；1→512 吞吐涨 464×、延迟不变 | 几乎必用 |
| 4 PagedAttention | KV 显存碎片 | 分页把利用率从 ~1% 拉到 ~98% | 几乎必用 |
| 5 前缀复用 | 重复 prefill | 长共享前缀省 63~86% TTFT；短前缀无用 | 有长共享前缀 |
| 6 量化 | 省显存/提速 | 显存永远省；decode 提速仅当 bandwidth-bound（大模型） | 显存紧 / 大模型 |
| 7 投机解码 | decode 串行 | 一次验多 token 几乎免费；无损 | 低并发、延迟敏感 |
| 8 服务指标 | 怎么衡量"快" | goodput = 满足 SLO 的吞吐；看 P99 不看平均 | —（方法论） |
| 9 引擎全景 | 落到 vLLM/SGLang/TRT | 地基相同，上层差异化 | —（综述） |

## 判断表一：瓶颈在哪

| 现象 | 瓶颈 | 典型场景 |
|------|------|----------|
| 卡在"搬权重"（算力闲） | **memory-bound** | decode、大模型、低 batch |
| 卡在"算"（算力满） | **compute-bound** | prefill、高 batch |
| 卡在"发指令"（GPU 等 CPU） | **launch-bound** | 小模型、低 batch、token 极少 |

判据：**算术强度**（FLOPs/搬运字节）低于 GPU 临界值 → memory-bound；高于 → compute-bound。

## 判断表二：什么场景上什么优化

| 场景特征 | 该上 | 不该上 |
|---------|------|--------|
| 高并发、离线、prompt 各异 | 大 batch + PagedAttention + 量化 | 投机解码、前缀复用 |
| 低并发、在线、延迟敏感 | 投机解码 + weight 量化 | —（大 batch 不适用） |
| 长共享 system prompt / 多轮 | 前缀复用（RadixAttention） | — |
| 显存 / 上下文 / 并发吃紧 | KV 量化 + GQA/MLA + 权重量化 | — |

> 通用地基（任何场景）：**PagedAttention + continuous batching**。

## 一句话记忆卡

- decode **慢** → batching（第3）、量化权重（第6）、投机解码（第7）。
- KV cache **吃显存** → PagedAttention（第4）、前缀复用（第5）、KV 量化/GQA（第2/6）。
- 怎么**衡量** → TTFT / TPOT / 吞吐 / **goodput** / SLO（第8）。
- 落到**引擎** → vLLM（通用）/ SGLang（前缀复用+可编程）/ TensorRT-LLM（编译极致）（第9）。

---

下一站：**第二部分 · GPU 架构与算子优化**——从"会用引擎"下钻到"读/写/优化 kernel"。
