# 学习路线图 (Roadmap)

> **总目标**：从算法工程师视角，吃透 LLM 的推理与训练 infra，最终能**优化框架提速**、
> **优化算子 (kernel)**，并能**高效地训练大模型**（把 MFU 拉满、把大规模训练跑稳）。
>
> 分六部分，由浅入深、前后依赖。这是一份**活的大纲**：随学习进度和兴趣随时调整。
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
| 6 | 量化 (INT8/INT4/FP8, GPTQ/AWQ, KV 量化) | 减少要搬的字节 → 提速省显存 | 🔜 |
| 7 | 投机解码 (Speculative / Medusa / EAGLE) | 用小模型/多头猜 token，绕开串行瓶颈 | ⬜ |
| 8 | 服务指标与压测 (TTFT/TPOT/吞吐/goodput/SLO) | 怎么量化"快"、怎么压测 | ⬜ |
| 9 | 推理引擎全景 (vLLM / SGLang / TensorRT-LLM) | 主流引擎架构对比 | ⬜ |

> **学完能**：看懂并配置一个高性能推理服务，知道每个旋钮为什么有效。

---

## 第二部分 · GPU 架构与算子优化 ★核心目标

**目标**：深入 GPU，学会读、写、优化一个算子 (kernel)。这是"如何优化算子"的正题。

| 章 | 主题 | 一句话学到什么 | 状态 |
|----|------|----------------|------|
| 10 | GPU 架构深入 | SM / warp / Tensor Core / roofline 模型 | ⬜ |
| 11 | CUDA 编程模型 | grid/block/thread、shared memory、访存合并 | ⬜ |
| 12 | 性能分析 | Nsight Systems/Compute、用 roofline 定位瓶颈 | ⬜ |
| 13 | 算子优化技术 | kernel fusion、tiling、occupancy、bank conflict | ⬜ |
| 14 | Triton 实战 | 手写高性能融合算子（fused softmax / layernorm） | ⬜ |
| 15 | FlashAttention 拆解 | IO-aware、online softmax、tiling；手撕简版 | ⬜ |
| 16 | 低精度算子 | Tensor Core、INT8/FP8 GEMM | ⬜ |

> **学完能**：读懂一个 CUDA/Triton 算子，能写并优化一个融合算子，会用 profiler 定位瓶颈。

---

## 第三部分 · 框架与编译优化 ★核心目标

**目标**：搞清楚框架层面怎么把模型跑快。这是"如何优化框架去提速"的正题。

| 章 | 主题 | 一句话学到什么 | 状态 |
|----|------|----------------|------|
| 17 | 计算图与图捕获 | eager vs graph、CUDA Graph 消除发射开销 | ⬜ |
| 18 | 编译栈 | torch.compile / TorchInductor 原理，XLA/TVM 概览 | ⬜ |
| 19 | 内存管理 | caching allocator、碎片、显存复用 | ⬜ |
| 20 | 调度与重叠 | 算子调度、H2D/D2H 重叠、多流 | ⬜ |
| 21 | 推理框架内部 | vLLM/SGLang 的 scheduler / block manager（源码级） | ⬜ |
| 22 | 端到端调优实战 | profile → 定位 → 优化 → 复测 | ⬜ |

> **学完能**：用 torch.compile / CUDA Graph / allocator 调优真实模型，并能读框架源码。

---

## 第四部分 · 分布式训练：并行与通信

**目标**：把训练扩展到多卡多机，掌握各种并行策略与它们背后的通信。

| 章 | 主题 | 一句话学到什么 | 状态 |
|----|------|----------------|------|
| 23 | 数据并行 (DP / DDP) | 梯度同步、bucketing、AllReduce | ⬜ |
| 24 | ZeRO 与 FSDP | 优化器状态/梯度/参数切分（1/2/3 级） | ⬜ |
| 25 | 张量并行 (TP) | 把单层矩阵切到多卡，机内 NVLink | ⬜ |
| 26 | 流水线并行 (PP) | 按层切分、micro-batch 与 bubble | ⬜ |
| 27 | 序列并行 / 上下文并行 / 3D 并行 | 长上下文切分与各并行怎么叠加 | ⬜ |
| 28 | 专家并行 (MoE) | expert parallel、all-to-all、负载均衡 | ⬜ |
| 29 | NCCL 与集合通信 | ring/tree AllReduce、通信原语 | ⬜ |
| 30 | 网络拓扑与通信调优 | NVLink/NVSwitch/InfiniBand/rail、NCCL 调参 | ⬜ |

> **学完能**：为一个大模型设计并行策略，判断瓶颈在通信还是计算，并会调通信。

---

## 第五部分 · 大模型训练工程与规模化 ★训练重点

**目标**：解决真正训大模型时天天要命的工程问题——显存、数据、容错、稳定性、效率、框架、集群。

| 章 | 主题 | 一句话学到什么 | 状态 |
|----|------|----------------|------|
| 31 | 训练显存预算与 offload | 手算 params+grads+optim+activations，recompute / CPU/ZeRO-offload | ⬜ |
| 32 | 混合精度与 FP8 训练 | bf16/fp16、loss scaling、Transformer Engine | ⬜ |
| 33 | 数据管线与 IO | TB 级加载、sequence packing、shuffle、吞吐、数据混配 | ⬜ |
| 34 | 分布式 checkpoint 与容错/弹性 | async/sharded ckpt、auto-resume、straggler、弹性训练 | ⬜ |
| 35 | 训练效率度量与扩展性 | MFU/HFU 怎么算、strong/weak scaling、瓶颈分析 | ⬜ |
| 36 | 训练稳定性与调试 | loss spike、NaN、梯度裁剪、确定性复现 | ⬜ |
| 37 | 训练框架内部 | Megatron-LM / DeepSpeed / torchtitan / NeMo | ⬜ |
| 38 | 集群、多机启动与可观测 | torchrun/rendezvous、SLURM/K8s、DCGM 监控、大规模 profiling | ⬜ |

> **学完能**：把一次大规模训练跑稳、跑快、跑省——OOM 会救、挂了能续、慢了能定位、MFU 能拉满。

---

## 第六部分 · 综合实战 (Capstone)

**目标**：把前五部分串起来，完成端到端的"定位瓶颈 → 落地提速"。

| 章 | 主题 | 状态 |
|----|------|------|
| 39 | 部署优化版推理服务（量化 + PagedAttention + continuous batching + CUDA Graph） | ⬜ |
| 40 | 端到端训练调优：把一个大模型的 MFU 拉满（并行 + FlashAttention + FP8 + torch.compile + 容错） | ⬜ |
| 41 | 自定义融合算子（Triton）集成进模型并测收益 | ⬜ |

> **学完能**：独立完成一次从瓶颈定位到落地提速的端到端优化（推理侧 + 训练侧）。

---

## 建议顺序与依赖

```
第一部分(推理) ─► 第二部分(算子) ─► 第三部分(框架) ─┐
                                                    ├─► 第六部分(实战)
第四部分(并行) ─► 第五部分(训练工程) ───────────────┘
```

- **默认路线**：一 → 二 → 三 → 四 → 五 → 六。先把推理主线走通（反馈直观），
  再深入算子和框架（两个核心目标），然后训练侧（并行 + 训练工程），最后实战收口。
- **训练优先路线**（你是做训练的，可选）：一(1–2) → 四(并行) → 五(训练工程) → 二/三(算子/框架) → 六。
  即先把训练这条线打通，再回头补算子和框架的底层功夫。
- 第二、三部分互为支撑，可交叉；第四、五部分是训练主线，建议连着学。

## 调整记录

- 2026-07-16：初版路线图（五部分）。
- 2026-07-16：面向大模型训练加厚——第四部分补 MoE 与网络拓扑/通信调优；
  新增第五部分「大模型训练工程与规模化」（显存预算、FP8、数据管线、容错 checkpoint、
  MFU/扩展性、稳定性调试、训练框架、集群可观测）；实战加"MFU 拉满"训练章。
