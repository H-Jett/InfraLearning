# 第二部分 · GPU 架构与算子优化

> 本部分导读。★ 这是你的核心目标之一：**优化算子 (kernel)**。

## 这一部分讲什么

第一部分我们学会了"用"推理引擎。但引擎的速度，归根到底来自它底下一个个**算子 (kernel)** 跑得够快。
这一部分带你**下钻到硬件和 kernel 层**：GPU 到底怎么执行、怎么读一个 kernel、怎么写一个 kernel、
怎么把它优化到接近硬件极限。

## 为什么算法工程师也要懂这一层

- **看懂**：FlashAttention、PagedAttention 这些"魔法"，本质都是 kernel 层的优化，懂了才不神秘；
- **会调**：训练/推理卡住时，能用 profiler 定位到是哪个 kernel、卡在算力还是带宽；
- **能写**：需要的算子框架没有、或不够快时，能自己用 Triton/CUDA 写一个融合算子（这正是第六部分实战）。

## 章节地图

- **第 10 章**：GPU 架构（SM/warp/Tensor Core/roofline/Brent's Theorem）——硬件地基。
- **第 11–13 章**：CUDA 编程模型、性能分析 (Nsight)、算子优化基本功（访存合并、bank conflict、
  向量化、register spill、warp divergence）。
- **第 14–15 章**：**手撕经典 kernel**——reduction/softmax/layernorm/elementwise（①），
  sgemm/transpose/scan/stream-compaction/top-k（②）。
- **第 16–17 章**：Triton 实战、**手撕 FlashAttention**。
- **第 18–19 章**：低精度算子 (INT8/FP8 GEMM)、Nvidia 现代 kernel 栈 (TMA/CUTLASS/CuTe)。

## 学习方式

比第一部分更"硬核"、更贴近底层，但仍是「概念 → 思考题 → 真机代码」。需要 GPU；会真的写和跑 kernel。

准备好了就从[第 10 章](10-gpu-arch.md)开始。
