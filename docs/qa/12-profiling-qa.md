# 第 12 章 · 思考题参考答案

> 只记录问题与参考答案。对应正文《第 12 章 12.7 思考题》。

<a id="q1"></a>

## 1. 从 `simt_sgemm` / `tensorop_gemm` 的名字能读出什么？为什么 bf16 快 3.5×？

**参考答案**：kernel 名直接暴露了走的**硬件路径**：
- `simt_sgemm` 的 **simt**（Single Instruction Multiple Threads）= 用**普通 CUDA core** 做的 FP32 矩阵乘；
- `tensorop_..._gemm` 的 **tensorop** = 用 **Tensor Core** 做的矩阵乘（`s16816` 是它一条指令处理的
  16×8×16 小矩阵乘加形状）。

bf16 快 3.5× 有两个原因（第 10 章）：① 走了 Tensor Core，专用矩阵乘单元吞吐远高于 CUDA core；
② bf16 是 2 字节、fp32 是 4 字节，搬运和存储都减半。实测 16.15 / 4.63 ≈ 3.5×，和第 10 章的峰值算力比一致。

<a id="q2"></a>

## 2. 一个 kernel 占了 90% 时间，下一步该看什么？

**参考答案**：先判断它是 **memory-bound 还是 compute-bound**——不同 bound 的优化方向完全相反，
不判断就优化等于瞎改。
- 有 ncu：看它的**访存吞吐 vs 计算吞吐**（占峰值百分比），哪个先顶到 100% 就是被哪个卡住；
- 没有 ncu：用第 10 章的**算术强度**估——算这个 kernel 的 FLOPs 和搬运字节，和 roofline 拐点比大小。

判断出来后对症：memory-bound → 访存合并 / tiling / 减字节；compute-bound → 上 Tensor Core / 低精度 /
减冗余计算。**先定位是哪种 bound，再开方。**

<a id="q3"></a>

## 3. ncu 更强的代价是什么？为什么生产常先用 nsys / torch.profiler？

**参考答案**：ncu 靠**读 GPU 硬件性能计数器 + 对目标 kernel 反复 replay** 来拿到 occupancy、访存合并率、
warp 停顿原因等细粒度指标——代价是：
1. **需要性能计数器权限**：容器/云环境默认常没开，一跑就 `ERR_NVGPUCTRPERM`；
2. **慢**：逐 kernel replay，profiling 一个真实负载可能比正常跑慢很多；
3. 一次聚焦少数 kernel，不适合先做全局扫描。

而 nsys / torch.profiler 只做**轻量 trace**（不碰性能计数器、不 replay），随处能跑、快，先用它们
**在全局层面找到大头 kernel**，再在有权限的机器上用 ncu 深钻那一个——由粗到细，效率最高。
