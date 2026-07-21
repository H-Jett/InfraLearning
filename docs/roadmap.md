# 学习路线图 (Roadmap)

> **总目标**：从算法工程师视角，吃透 LLM 的推理与训练 infra，最终能**优化框架提速**、
> **优化算子 (kernel)**，并能**高效地训练大模型**（把 MFU 拉满、把大规模训练跑稳）。
>
> 分七部分，由浅入深、前后依赖。这是一份**活的大纲**：随学习进度和兴趣随时调整。
> 状态标记：✅ 已完成 · 🔜 进行中 · ⬜ 待学。

---

## 第一部分 · 推理与服务

**目标**：理解推理系统怎么跑、瓶颈在哪、业界怎么把它做快。这是全书的地基。

| 章 | 主题 | 一句话学到什么 | 状态 |
|----|------|----------------|------|
| 1 | Prefill 与 Decode | 推理两阶段的不对称，memory-bound 的本质 | ✅ |
| — | 补充 A：GPU 存储层级与 HBM | "从显存搬权重"的硬件背景 | ✅ |
| — | 补充 B：CUDA 异步/同步与基准测试 | 怎么在 GPU 上准确计时 | ✅ |
| 2 | KV Cache：原理与显存计算 | 缓存了什么、显存怎么一字节字节算 | ✅ |
| 3 | Batching：static → continuous batching | 吞吐第一杠杆，GPU 怎么喂饱 | ✅ |
| 4 | PagedAttention 与显存管理 | 像操作系统分页一样管 KV cache | ✅ |
| 5 | 前缀复用 (Prefix Caching / RadixAttention) | 共享前缀怎么省算力和显存 | ✅ |
| 6 | 量化 (INT8/INT4/FP8, GPTQ/AWQ, KV 量化) | 减少要搬的字节 → 提速省显存 | ✅ |
| 7 | 投机解码 (Speculative / Medusa / EAGLE) | 用小模型/多头猜 token，绕开串行瓶颈 | ✅ |
| 8 | 服务指标与压测 (TTFT/TPOT/吞吐/goodput/SLO) | 怎么量化"快"、怎么压测 | ✅ |
| 9 | 推理引擎全景 (vLLM / SGLang / TensorRT-LLM) | 主流引擎架构对比 | ✅ |

> **学完能**：看懂并配置一个高性能推理服务，知道每个旋钮为什么有效。

---

## 第二部分 · GPU 架构与算子优化 ★核心目标

**目标**：深入 GPU，学会读、写、优化一个算子 (kernel)——含大量"手撕 kernel"实战。这是"如何优化算子"的正题。

| 章 | 主题 | 一句话学到什么 | 状态 |
|----|------|----------------|------|
| 10 | GPU 架构深入 | SM / warp / Tensor Core / roofline / Brent's Theorem | ✅ |
| 11 | CUDA 编程模型 | grid/block/thread、shared memory、访存合并 | 🔜 |
| 12 | 性能分析 | Nsight Systems/Compute、用 roofline 定位瓶颈 | ⬜ |
| 13 | 算子优化基本功 | bank conflict + padding/swizzling、occupancy、向量化读取、register spill、warp divergence | ⬜ |
| 14 | 手撕经典 kernel ① | reduction / softmax / layernorm / elementwise（消除 warp divergence） | ⬜ |
| 15 | 手撕经典 kernel ② | sgemm（tiling/split-K）、transpose、scan / stream compaction、top-k（bitonic） | ⬜ |
| 16 | Triton 实战 | 手写高性能融合算子 | ⬜ |
| 17 | FlashAttention 拆解 | IO-aware、online softmax、tiling；三个版本；手撕简版 | ⬜ |
| 18 | 低精度算子 | Tensor Core、INT8/FP8 GEMM | ⬜ |
| 19 | Nvidia 现代 kernel 栈 | TMA、CUTLASS、CuTe DSL、warp specialization | ⬜ |

> **学完能**：读懂并手写/优化一个 CUDA/Triton 算子（reduction/gemm/scan/topk…），会用 profiler 定位瓶颈。

---

## 第三部分 · 框架与编译优化 ★核心目标

**目标**：搞清楚框架层面怎么把模型跑快。这是"如何优化框架去提速"的正题。

| 章 | 主题 | 一句话学到什么 | 状态 |
|----|------|----------------|------|
| 20 | 计算图与图捕获 | eager vs graph、CUDA Graph 消除发射开销 | ⬜ |
| 21 | 编译栈 | torch.compile / TorchInductor 原理，XLA/TVM 概览 | ⬜ |
| 22 | 内存管理 | caching allocator、碎片、显存复用 | ⬜ |
| 23 | 调度与重叠 | 算子调度、H2D/D2H 重叠、多流 | ⬜ |
| 24 | 推理框架内部 | vLLM/SGLang 的 scheduler / block manager（源码级） | ⬜ |
| 25 | 端到端调优实战 | profile → 定位 → 优化 → 复测 | ⬜ |

> **学完能**：用 torch.compile / CUDA Graph / allocator 调优真实模型，并能读框架源码。

---

## 第四部分 · 分布式训练：并行与通信

**目标**：把训练扩展到多卡多机，掌握各种并行策略与它们背后的通信。

| 章 | 主题 | 一句话学到什么 | 状态 |
|----|------|----------------|------|
| 26 | 数据并行 (DP / DDP) | 梯度同步、bucketing、AllReduce | ⬜ |
| 27 | ZeRO 与 FSDP | 优化器状态/梯度/参数切分（1/2/3 级） | ⬜ |
| 28 | 张量并行 (TP) | 把单层矩阵切到多卡，机内 NVLink | ⬜ |
| 29 | 流水线并行 (PP) | 按层切分、micro-batch 与 bubble | ⬜ |
| 30 | 序列并行 / 上下文并行 / 3D 并行 | 长上下文切分与各并行怎么叠加 | ⬜ |
| 31 | 专家并行 (MoE) | expert parallel、all-to-all、负载均衡 | ⬜ |
| 32 | NCCL 与集合通信 | ring/tree AllReduce、通信原语 | ⬜ |
| 33 | 网络拓扑与通信调优 | NVLink/NVSwitch/InfiniBand/rail、NCCL 调参、NVSHMEM/PGAS | ⬜ |

> **学完能**：为一个大模型设计并行策略，判断瓶颈在通信还是计算，并会调通信。

---

## 第五部分 · 大模型训练工程与规模化 ★训练重点

**目标**：解决真正训大模型时天天要命的工程问题——显存、数据、容错、稳定性、效率、框架、集群。

| 章 | 主题 | 一句话学到什么 | 状态 |
|----|------|----------------|------|
| 34 | 训练显存预算与 offload | 手算 params+grads+optim+activations，recompute / CPU/ZeRO-offload | ⬜ |
| 35 | 混合精度与 FP8 训练 | bf16/fp16、loss scaling、Transformer Engine | ⬜ |
| 36 | 数据管线与 IO | TB 级加载、sequence packing、shuffle、吞吐、数据混配 | ⬜ |
| 37 | 分布式 checkpoint 与容错/弹性 | async/sharded ckpt、auto-resume、straggler、弹性训练 | ⬜ |
| 38 | 训练效率度量与扩展性 | MFU/HFU 怎么算、strong/weak scaling、瓶颈分析 | ⬜ |
| 39 | 训练稳定性与调试 | loss spike、NaN、梯度裁剪、确定性复现 | ⬜ |
| 40 | 训练框架内部 | Megatron-LM / DeepSpeed / torchtitan / NeMo | ⬜ |
| 41 | 集群、多机启动与可观测 | torchrun/rendezvous、SLURM/K8s、DCGM 监控、大规模 profiling | ⬜ |

> **学完能**：把一次大规模训练跑稳、跑快、跑省——OOM 会救、挂了能续、慢了能定位、MFU 能拉满。

---

## 第六部分 · 综合实战 (Capstone)

**目标**：把前五部分串起来，完成端到端的"定位瓶颈 → 落地提速"。

| 章 | 主题 | 状态 |
|----|------|------|
| 42 | 部署优化版推理服务（量化 + PagedAttention + continuous batching + CUDA Graph） | ⬜ |
| 43 | 端到端训练调优：把一个大模型的 MFU 拉满（并行 + FlashAttention + FP8 + torch.compile + 容错） | ⬜ |
| 44 | 自定义融合算子（Triton）集成进模型并测收益 | ⬜ |

> **学完能**：独立完成一次从瓶颈定位到落地提速的端到端优化（推理侧 + 训练侧）。

---

## 第七部分 · 拓展与面试

**目标**：补齐 AI infra 工程师的"周边"广度与面试硬功——独立成章，作为主线之外的拓展。

| 章 | 主题 | 一句话学到什么 | 状态 |
|----|------|----------------|------|
| 45 | C++ 与工程基础 | const pointer / 虚函数继承、模板元编程、Cmake / 动静态链接、常用设计模式 | ⬜ |
| 46 | OS 与网络基础 | write-through/back、memory fence、heap/stack、lock-free、内核态/用户态、zero-copy、五层网络、三次握手 | ⬜ |
| 47 | ML / CV 速览（拓展） | RNN/Transformer/attention；CNN 复杂度/ResNet/ViT/U-Net；forward-backward、MLP 手写 | ⬜ |
| 48 | AI infra 面试专题 | 快问快答清单 + 手撕 kernel 题集（LeetGPU/DeepML）+ 高频考点复盘 | ⬜ |

> **学完能**：应对 AI infra 岗位面试的广度问答与手撕环节；理解 infra 依赖的底层 CS / C++ 概念。

---

## 建议顺序与依赖

```
第一部分(推理) ─► 第二部分(算子) ─► 第三部分(框架) ─┐
                                                    ├─► 第六部分(实战)
第四部分(并行) ─► 第五部分(训练工程) ───────────────┘
第七部分(拓展与面试)：随时穿插，或临面试前集中过
```

- **默认路线**：一 → 二 → 三 → 四 → 五 → 六；第七部分作为拓展随时穿插。
- **训练优先路线**（你是做训练的，可选）：一(1–2) → 四(并行) → 五(训练工程) → 二/三(算子/框架) → 六。
- 第二、三部分互为支撑，可交叉；第四、五部分是训练主线，建议连着学。

## 调整记录

- 2026-07-16：初版（五部分）。
- 2026-07-16：面向大模型训练加厚——第四部分补 MoE 与通信调优；新增第五部分「训练工程与规模化」；实战加 MFU 章。
- 2026-07-21：依据一份 AI infra 面试大纲——第二部分加厚（手撕 kernel ①②、Nvidia 现代 kernel 栈 TMA/CUTLASS/CuTe、Brent's Theorem）；新增第七部分「拓展与面试」（C++/工程、OS/网络、ML/CV 速览、面试专题）。
