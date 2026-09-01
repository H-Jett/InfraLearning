# 第 12 章 性能分析：用 profiler 找瓶颈

> 本章目标：学会用 profiler **不靠猜**地定位性能问题——哪个 kernel 在耗时、它卡在访存还是计算。
> 介绍三层 profiler（torch.profiler / Nsight Systems / Nsight Compute），并用真机数据演示。

## 12.1 别猜，去测

优化的第一条纪律：**先测，再改**。凭直觉猜"哪里慢"经常错——真正的瓶颈可能是一个你没注意的
逐元素算子、一次没必要的同步、或一个走错了硬件路径的 kernel。profiler 就是把这些**摊开给你看**的工具。

## 12.2 三层 profiler，各看一个粒度

| 工具 | 粒度 | 回答什么 | 是否需要特殊权限 |
|------|------|---------|----------------|
| **torch.profiler** | 框架 / kernel 级 | 哪个算子、哪个 kernel 占了多少时间 | 否（最易用，随处可跑） |
| **Nsight Systems** (`nsys`) | 系统时间线 | CPU↔GPU 怎么交互、kernel 怎么排布、有没有空隙 | 否（只做 trace） |
| **Nsight Compute** (`ncu`) | 单 kernel 显微镜 | 这个 kernel 的占用率、访存/计算吞吐、合并率 | **是**（读硬件性能计数器） |

**从粗到细**：先用 torch.profiler / nsys 找到"哪个 kernel 是大头"，再用 ncu 钻进那个 kernel 看它为什么慢。

## 12.3 torch.profiler：找出主 kernel（真机）

最易上手。用它抓同一个 8192³ 矩阵乘、fp32 与 bf16 两种精度，看**最耗时的 GPU kernel 是谁**：

| 精度 | 每次调用 | 主 kernel（名字截断） |
|------|--------:|----------------------|
| fp32 | 16.15 ms | `cutlass_80_simt_sgemm_256x128...` |
| bf16 | 4.63 ms | `cutlass_80_tensorop_bf16_s16816gemm...` |

**光看 kernel 名就读出了硬件路径**：

- `simt_sgemm` 里的 **simt = CUDA core**（第 10 章的 FP32 通路）；
- `tensorop_..._gemm` 里的 **tensorop = Tensor Core**（`s16816` 是它的 16×8×16 矩阵乘加指令形状）。

于是第 10 章"fp32 走 CUDA core、bf16 走 Tensor Core、快 3.5×"的结论，**不靠猜，用 profiler 直接验出来了**
（16.15 / 4.63 ≈ 3.5×）。这就是 profiler 的价值：**告诉你时间花在哪个 kernel、它用了什么硬件**。

> 用法要点：`with profile(activities=[ProfilerActivity.CUDA]) as prof: …`，再
> `prof.key_averages().table(sort_by="cuda_time_total")`；计时前**务必 warmup + synchronize**（第 1 章补充 B）。

## 12.4 Nsight Systems：看时间线

`nsys profile --stats=true python xxx.py` 会给一条**系统时间线**和汇总表。对上面同一个 fp32 matmul，
它的 `cuda_gpu_kern_sum` 报告显示：

```
Time(%)  Total(ns)  Instances  ...  Name
  99.1    6052658       3       ...  cutlass_80_simt_sgemm_256x128_8x4_nn_align1
```

nsys 擅长回答 torch.profiler 看不清的**系统层问题**：kernel 之间有没有空隙（GPU 在等 CPU？）、
`cudaMemcpy` 挡住了没、多个流有没有重叠、launch 开销占比多少（第 1 章的 launch-bound 就靠它坐实）。

## 12.5 Nsight Compute：单 kernel 的显微镜（+ 一个真实的坑）

`ncu` 深入**单个 kernel**，读硬件性能计数器，给出 torch.profiler 给不了的东西：

- **占用率 (occupancy)**[^occ]：SM 上活跃 warp 占最大可能的比例——越高越能用别的 warp 掩盖访存延迟；
- **访存 / 计算吞吐**（占峰值的百分比）：直接告诉你这个 kernel 是 memory-bound 还是 compute-bound；
- **访存合并情况**：每次访存请求实际搬了多少有用字节（合并好 vs 差，第 11 章那个 5.5× 差距，ncu 能量化）；
- **warp 停顿原因**：卡在等访存、等依赖、还是等发射。

> **真实的坑**：ncu 要读 GPU 性能计数器，很多容器 / 云环境默认没开这个权限，一跑就报
> **`ERR_NVGPUCTRPERM`**（本书写作的机器就是）。解决要管理员层面开启（如容器加 `--cap-add`、
> 或设 `NVreg_RestrictProfilingToAdminUsers=0`）。**所以生产排查常先靠 nsys + torch.profiler**，
> 拿到 ncu 权限的机器再做深钻。

## 12.6 定位瓶颈的方法论

把三层串起来，标准流程是：

1. **torch.profiler / nsys**：按 GPU 时间排序，找到**占大头的那个 kernel**（别在小头上浪费精力）；
2. **判断它 memory-bound 还是 compute-bound**：有 ncu 就看访存/计算吞吐；没有就用第 10 章的算术强度估；
3. **对症下药**：
   - memory-bound → 访存合并（第 11 章）、tiling 提高算术强度、量化减字节、算子融合；
   - compute-bound → 换更快的硬件路径（Tensor Core / 低精度）、减少冗余计算；
   - launch-bound（小 kernel 太多）→ 算子融合、CUDA Graph（第三部分）。

**profiler 负责"定位"，前几章的知识负责"开方"。**

## 12.7 思考题

1. profiler 显示 fp32 matmul 的主 kernel 叫 `..._simt_sgemm`、bf16 的叫 `..._tensorop_..._gemm`。
   光看名字你能读出什么？为什么 bf16 那个快 3.5×？
2. 你用 profiler 抓到某个 kernel 占了总时间的 90%。**下一步**该看什么，才知道该怎么优化它？
3. ncu 能给出 occupancy、访存合并率这些 torch.profiler 给不了的信息，代价是什么？
   为什么生产排查常常先用 nsys / torch.profiler？

> 📖 **参考答案**（想清楚再看）：[Q1](../../qa/12-profiling-qa.md#q1) · [Q2](../../qa/12-profiling-qa.md#q2) · [Q3](../../qa/12-profiling-qa.md#q3)

## 12.8 延伸阅读

- 第 10 章 roofline / 第 11 章 访存合并：profiler 定位到瓶颈后，用它们判断和优化。
- 第 13 章 算子优化基本功：ncu 里 occupancy、bank conflict、register spill 这些指标怎么改善。
- 第三部分 · 框架优化：CUDA Graph、torch.compile 针对 launch-bound / 调度问题。

## 12.9 附录：本章练习代码

源码：`exercises/02-gpu-kernels/12_profiling.py`。

<!-- CODE:exercises/02-gpu-kernels/12_profiling.py START -->
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 12 章练习：用 torch.profiler 找瓶颈——看清"哪个 kernel 在耗时、走的哪条硬件路径"。

我们对同一个矩阵乘用 fp32 和 bf16 两种精度各跑一遍，用 profiler 抓出：
  - 占用 GPU 时间最多的 kernel 是谁（名字里能看出走 CUDA core 还是 Tensor Core）；
  - 每次调用耗时多少。
这正是第 10 章"fp32 用 CUDA core、bf16 用 Tensor Core"的结论——用 profiler 亲手验出来。
"""

import torch
from torch.profiler import profile, ProfilerActivity


def cuda_us(e):
    # 兼容不同 torch 版本的字段名
    return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)


def top_kernel(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    ka = sorted(prof.key_averages(), key=cuda_us, reverse=True)
    for e in ka:
        if cuda_us(e) > 0:
            return e.key, cuda_us(e) / 1e3 / iters  # ms/call
    return "?", 0.0


def main():
    N = 8192
    a32 = torch.randn(N, N, device="cuda"); b32 = torch.randn(N, N, device="cuda")
    a16 = a32.to(torch.bfloat16); b16 = b32.to(torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = False  # 让 fp32 走真正的 CUDA core

    print("用 torch.profiler 抓 8192³ 矩阵乘，最耗时的 GPU kernel：\n")
    for name, fn in [("fp32", lambda: torch.mm(a32, b32)),
                     ("bf16", lambda: torch.mm(a16, b16))]:
        k, ms = top_kernel(fn)
        print(f"[{name}] {ms:6.2f} ms/call")
        print(f"       主 kernel: {k[:72]}\n")

    print("读法：profiler 直接告诉你时间花在哪个 kernel 上；kernel 名里的 simt=CUDA core、")
    print("      含 tensor/wgmma/hmma 等字样=Tensor Core。这样不必猜，一眼看出走了哪条硬件路径。")


if __name__ == "__main__":
    main()
```
<!-- CODE:exercises/02-gpu-kernels/12_profiling.py END -->

**运行输出：**

<!-- OUTPUT:exercises/02-gpu-kernels/outputs/12_profiling.txt START -->
```text
用 torch.profiler 抓 8192³ 矩阵乘，最耗时的 GPU kernel：

[fp32]  15.98 ms/call
       主 kernel: void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>(cutla

[bf16]   4.63 ms/call
       主 kernel: void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x

读法：profiler 直接告诉你时间花在哪个 kernel 上；kernel 名里的 simt=CUDA core、
      含 tensor/wgmma/hmma 等字样=Tensor Core。这样不必猜，一眼看出走了哪条硬件路径。
```
<!-- OUTPUT:exercises/02-gpu-kernels/outputs/12_profiling.txt END -->

---

[^occ]: Occupancy（占用率）：SM 上活跃 warp 数占最大可能的比例，受每 block 的寄存器/shared memory 用量限制。详见[术语表](../../glossary.md#occupancy)。
