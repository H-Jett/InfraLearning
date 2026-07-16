# 第 1 章 问答记录

## 提问：为什么 decode 是访存受限 (memory-bound) 的？

**答（核心）**：用算术强度 (arithmetic intensity = FLOPs / 搬运字节数) 判断。

一次运算耗时 ≈ max(FLOPs/算力, 字节/带宽)。矩阵乘 `W·x`：

- **Prefill**（一次 N 个 token，输入 `[N, d_in]`）：FLOPs≈`2N·d_out·d_in`，搬 W 字节≈`2·d_out·d_in`，
  算术强度 ≈ **N**（几百上千）→ 高于临界强度 → **compute-bound**。
- **Decode**（batch=1，输入 `[1, d_in]`）：FLOPs≈`2·d_out·d_in`，搬 W 字节≈`2·d_out·d_in`，
  算术强度 ≈ **1** → 远低于现代 GPU 临界强度(约 100~300) → **memory-bound**。

**本质**：每生成 1 个 token，都要把整个模型权重从 HBM 完整读一遍；每个权重只用 1 次乘加就丢，算力空转。

**能记一辈子的估算**：TPOT 理论下限 ≈ 模型大小 ÷ 显存带宽。
- 70B fp16 = 140GB，带宽 ~2TB/s → 下限 ≈ 70ms/token（算力再强也突破不了 → 量化的动机）。
- 0.6B = 1.2GB → 下限 <1ms，但实测 37.9ms → **小模型 batch=1 时真正瓶颈是 kernel 发射等固定开销**，
  还没轮到带宽。memory-bound 是原理；小模型上固定开销更抢戏（诚实结论）。

---

## Q1：8 个请求一起 decode，总耗时接近 8×37.9ms 还是 1×37.9ms？

**我的答**：接近 1×，因为 batch 只需一次显存搬运。

**判定：正确 ✅**（抓住了 batching 是吞吐第一杠杆的根本原因）。

**补充严谨性**：
1. 不是精确 1×。搬权重成本共享(几乎不变)，但计算量是 8×；因 batch=1 时算力闲置，这 8× 计算
   "填进空闲几乎免费"，直到 batch 把算力也填满，从 memory-bound 翻转为 compute-bound，此后线性变慢。
2. 每条序列的 KV cache 不共享，attention 段读取随 batch 增长。
3. 结论：远小于 8×、接近 1×，但存在"甜点 batch size"，超过后收益递减 → 引出第 3 章 continuous batching。

---

## Q2：为什么 16 token 的 prefill(52ms) 比 256 token(33ms) 还慢？

**我的疑惑**：不理解。

**答**：别读成"越短越慢"的趋势。16 与 256 都在 **overhead-bound（固定开销主导）** 区间：
一次前向要发射几十上百个 CUDA kernel + Python 调度，这些开销固定、与 token 数无关；而 16 vs 256 个
token 的实际计算对 5090 都小到忽略，藏在开销下。所以此区间总耗时近似平线(30~50ms)，
52 vs 33 是噪声/预热抖动，不是结构性趋势。

真正信号：per-token 耗时持续暴跌(并行度高)；总耗时到 ≥1024 才随长度真正爬升(进入 compute-bound)。
**教训：小规模微基准看趋势，别看单点。**

---

## 追问 1：“搬运”是把参数从显存搬到计算单元吗？

**答**：方向对，但要穿过存储层级，不是一跳：
```
HBM 显存(大, ~1.8TB/s) → L2(几十MB) → shared mem/L1/寄存器(几百KB, 极快) → 计算单元
```
计算单元只能吃寄存器里的数据；权重几个 GB，片上内存装不下，只能常驻 HBM，每次算都要从 HBM 流上来。
**瓶颈精确在 `HBM → 片上` 这一跳**（HBM 带宽最低、权重最大）。所以 decode memory-bound 更准确叫
**HBM 带宽受限**。推论：减少要搬的字节(量化 fp16→int8/int4)能直接提速——这堵墙矮了。

## 追问 2：Q2 是说 token 小时耗时不在计算、而在固定启动开销吗？

**答：正确 ✅**。固定开销 = ① CUDA kernel 发射延迟(每算子一个 kernel，一次前向发几十上百个)；
② Python/框架逐算子调度。token 少时 GPU 算得比 CPU 喂得快，卡在“发指令”而非“算”，
行话 **launch-bound / CPU-bound**。真实引擎用 **CUDA Graph**(录制整串 kernel 一次回放)、
**kernel 融合 / torch.compile** 来砍这个开销。

**两种“喂不饱”要分清**：
| 区间 | 卡在哪 | 名字 |
|---|---|---|
| decode/batch=1、大模型 | 从 HBM 搬权重 | memory-bound(带宽墙) |
| token 很少、小模型 | CPU 发 kernel 指令 | launch-bound(开销墙) |
