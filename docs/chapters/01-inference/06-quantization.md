# 第 6 章 量化：省显存与提速

> 本章目标：讲清量化为什么能省显存、什么时候才真的能提速，以及量化"权重 / KV cache / 激活"
> 各自省的是哪一项。这一课也是训练侧（FP8 训练、混合精度）的铺垫。数据用 Qwen3-0.6B 真机实测。

## 6.1 量化是什么，为什么要做

**量化 (quantization)**[^quant] = 用更少的比特表示数值。把权重 / 激活 / KV 从 bf16（2 字节）
压到 int8（1 字节）、int4（0.5 字节）甚至 fp8。三个收益：

1. **省显存**：字节数直接按比例下降（权重、KV cache 都能省）；
2. **提速（有条件）**：decode 是访存受限时，搬的字节少了，TPOT 下限就低了（第 1 章公式）；
3. **低精度算力**：int8 / fp8 的 Tensor Core 吞吐更高，能加速 compute-bound 的计算。

## 6.2 量化的基本原理

把一段浮点数线性映射到整数范围：

```
q = round(x / scale) + zero_point      # 量化
x ≈ (q - zero_point) × scale           # 反量化
```

- **scale**（缩放）决定浮点范围怎么塞进整数格子；**zero_point**（零点）用于非对称量化；
- 量化粒度：**per-tensor**（整个张量一个 scale，最粗）→ **per-channel**（每列一个）→
  **per-group**（每 128 个元素一个，最细）。越细，精度损失越小、开销越大；
- 代价是**量化误差**——精度越低、粒度越粗，误差越大。

## 6.3 量化什么？各省哪一项（对上前几章的公式）

| 量化对象 | 省的是 | 对应公式 | 主要收益 |
|---------|-------|---------|---------|
| **权重** | 权重显存 | 第 1 章 TPOT 下限 = 模型字节/带宽 | 省显存 + 给大模型 decode 提速 |
| **KV cache** | KV 显存 | 第 2 章 KV_bytes 里的"每元素字节" | 同显存塞更多并发 / 更长上下文 |
| **激活** | 计算精度 | —（用 int8/fp8 Tensor Core） | 加速 compute-bound 的 prefill |

这张表是本章的骨架：**量化不是笼统地"变快变小"，而是精准地砍某一项。**

## 6.4 常见精度格式

- **fp32**（4 字节）：训练/推理的老基线，现在很少全程用。
- **bf16 / fp16**（2 字节）：当前主流基线。bf16 动态范围大、更稳，训练常用。
- **fp8**（1 字节，E4M3 / E5M2）：新硬件（Hopper 起）原生支持，训练推理都在用。
- **int8**（1 字节）/ **int4**（0.5 字节）：推理量化主力，int4 多用于 weight-only。

## 6.5 实验 A：显存精确减半，但速度几乎没变（关键！）

fp32 vs bf16，测权重显存和 decode 每步耗时（Qwen3-0.6B，batch=1）：

| 精度 | 权重显存 | decode 每步 |
|------|--------:|-----------:|
| fp32 | 2274 MiB | 17.47 ms |
| bf16 | 1137 MiB | 16.81 ms |
| **比值** | **2.00×** | **1.04×** |

**显存精确减半，速度却几乎没动。** 为什么？回到第 1 章的"两种喂不饱"：

> 0.6B 小模型 + batch=1 的 decode 是 **launch-bound**（卡在 kernel 发射开销），**不是**显存带宽受限。
> 既然瓶颈不在"搬权重"，那把权重字节减半自然救不了速度——**但显存实打实省了一半**。

这是量化最容易被误解的地方：**"量化一定更快"是错的**。省显存永远成立；提不提速，要看 decode 到底卡在哪。

## 6.6 实验 B：什么时候降精度才真的提速

用第 1 章的公式 **TPOT 下限 ≈ 模型字节 / 显存带宽**（设带宽 ≈ 1.8 TB/s）算一算：

| 模型 | 精度 | 权重大小 | TPOT 下限 | 状态 |
|------|-----|--------:|---------:|------|
| 0.6B | bf16 | 1.2 GB | 0.67 ms | 下限 ≪ 实测 17 ms → **launch-bound**，降精度不提速 |
| 0.6B | int4 | 0.3 GB | 0.17 ms | 同上 |
| 70B  | bf16 | 140 GB | 77.8 ms | 下限就是几十 ms 量级 → **bandwidth-bound** |
| 70B  | int8 | 70 GB  | 38.9 ms | 按字节比 → decode ~2× |
| 70B  | int4 | 35 GB  | 19.4 ms | 按字节比 → decode ~4× |

**结论**：
- **小模型 / 低 batch**：decode 是 launch-bound，量化只省显存、几乎不提速；
- **大模型**：decode 是 bandwidth-bound，量化权重能让 decode **按字节比例线性提速**（70B int4 ≈ 4× decode）。

这就精确回答了"量化到底能不能提速"：**取决于 decode 是不是真的卡在带宽上。**

## 6.7 两条技术路线：weight-only vs W8A8

- **Weight-only（如 W4A16）**：只把**权重**压成 int4，计算时反量化回 fp16 再算。
  - 加速**大模型的 decode**（memory-bound：加载的权重字节少了 4×）；
  - 对 **prefill 帮助不大**（prefill 是 compute-bound，计算还是 fp16，没变快）；
  - 代表方法：**GPTQ**[^gptq]、**AWQ**[^awq]（都是训练后量化，用少量校准数据，int4 常近乎无损）。
- **W8A8（权重+激活都 int8）**：权重和激活都量化，用 **int8 Tensor Core** 算。
  - 连 **compute-bound 的 prefill 也能提速**；
  - 代表方法：**SmoothQuant**（把激活的"异常值"难题转移到权重上，让激活也好量化）。

## 6.8 训练侧的量化（第五部分预告）

量化不只属于推理：

- **混合精度训练**：用 bf16/fp16 算、fp32 存主权重，省显存又稳（第 32 章）；
- **FP8 训练**：新硬件用 fp8 做 GEMM，进一步降本，靠 **Transformer Engine** 等库处理缩放和数值稳定
  （第 32 章）。

**核心权衡都一样**：用精度换显存 / 速度，代价是数值误差——训练比推理对误差更敏感，所以更讲究。

## 6.9 思考题

1. 实验 A 里 fp32→bf16 显存精确减半，但 decode 速度几乎没变。为什么？
   什么情况下量化才会真正加速 decode？
2. 你要把一个模型能服务的**最大上下文 / 并发数**提上去（KV cache 吃紧），
   应该优先量化"权重"还是"KV cache"？为什么？
3. weight-only 量化（W4A16：权重压 int4，计算仍 fp16）主要加速推理的哪个阶段——prefill 还是 decode？
   为什么它对另一个阶段帮助不大？

> 参考答案见仓库 `qa/06-quantization-qa.md`。

## 6.10 延伸阅读

- 第 8 章 服务指标：量化的收益（显存/延迟/吞吐）怎么在压测里量出来。
- 第二部分 · 算子优化：int8/fp8 GEMM、Tensor Core 是量化提速的底层（第 16 章）。
- 第五部分 · 训练工程：混合精度与 FP8 训练（第 32 章）。

## 6.11 附录：本章练习代码

由 `scripts/sync_code.py` 从源码自动同步。源码：`exercises/01-inference/06_quantization.py`。

<!-- CODE:exercises/01-inference/06_quantization.py START -->
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 6 章练习：把"精度 → 字节数 → 显存 / 速度"这条链量化出来，并搞清楚
"量化到底什么时候才真的能给 decode 提速"。

实验 A（真机）：fp32 vs bf16，测权重显存和 decode 每步耗时。
    预期：显存精确减半；但小模型、低 batch 时 decode 速度几乎不变——因为它是
    launch-bound（卡在 kernel 发射开销，不是带宽），减字节救不了。呼应第 1 章。

实验 B（理论）：用第 1 章的公式 "TPOT 下限 ≈ 模型字节 / 显存带宽" 说明——
    只有当 decode 真正 bandwidth-bound（大模型）时，降精度才线性提速；
    并列出权重 / KV cache 在各精度下的显存。

绝对路径：
  /volume/data/hjiang02/workspace/infra-learning/exercises/01-inference/06_quantization.py
"""

import time
import statistics
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/volume/data/models/Qwen3-0.6B"
DEVICE = "cuda:0"
BW = 1.8e12  # 假设显存带宽 ~1.8 TB/s（数量级示意）


def sync():
    torch.cuda.synchronize()


def bench(dtype, bs, base_ids):
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=dtype).to(DEVICE).eval()
    mem = sum(p.numel() * p.element_size() for p in m.parameters()) / 2**20
    ids = base_ids.repeat(bs, 1)

    @torch.no_grad()
    def dec(n):
        o = m(ids, use_cache=True); p = o.past_key_values; x = o.logits[:, -1:].argmax(-1)
        sync(); t0 = time.perf_counter()
        for _ in range(n):
            o = m(x, past_key_values=p, use_cache=True); p = o.past_key_values; x = o.logits[:, -1:].argmax(-1)
        sync(); return time.perf_counter() - t0

    with torch.no_grad():
        dec(4)
        t = statistics.median([dec(64) for _ in range(3)])
    del m; torch.cuda.empty_cache()
    return mem, t / 64 * 1000


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = tok("请简要介绍一下你自己。", return_tensors="pt").input_ids.to(DEVICE)

    # ---------- 实验 A ----------
    print("=" * 68)
    print("实验 A（真机）：fp32 vs bf16 —— 显存减半，但速度几乎不变（为什么？）")
    print("=" * 68)
    print(f"{'精度':>6} | {'权重显存(MiB)':>14} | {'decode每步(ms)':>16}")
    print("-" * 44)
    res = {}
    for dtype, name in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
        mem, ms = bench(dtype, 1, base)
        res[name] = (mem, ms)
        print(f"{name:>6} | {mem:>14.0f} | {ms:>16.2f}")
    print(f"\n显存比 fp32/bf16 = {res['fp32'][0]/res['bf16'][0]:.2f}x（精确减半）")
    print(f"速度比 fp32/bf16 = {res['fp32'][1]/res['bf16'][1]:.2f}x（几乎没变！）")
    print("原因：0.6B 小模型 + batch=1 的 decode 是 launch-bound（卡在 kernel 发射开销，")
    print("      不是显存带宽），所以减少权重字节救不了速度——但显存实实在在省了一半。")

    # ---------- 实验 B ----------
    print("\n" + "=" * 68)
    print("实验 B（理论）：什么时候降精度才真的提速？TPOT 下限 ≈ 模型字节 / 带宽")
    print("=" * 68)
    print(f"（假设显存带宽 ≈ {BW/1e12:.1f} TB/s）\n")
    print(f"{'模型':>8} | {'精度':>5} | {'权重大小':>10} | {'TPOT下限(ms)':>13} | 说明")
    print("-" * 68)
    for pname, params in [("0.6B", 0.6e9), ("70B", 70e9)]:
        for prec, bytes_per in [("bf16", 2), ("int8", 1), ("int4", 0.5)]:
            size = params * bytes_per
            floor_ms = size / BW * 1000
            note = ""
            if pname == "0.6B":
                note = "下限<<实测17ms → launch-bound，降精度不提速"
            else:
                note = "下限就是几十ms量级 → bandwidth-bound，降精度≈线性提速"
            print(f"{pname:>8} | {prec:>5} | {size/1e9:>8.1f}GB | {floor_ms:>13.2f} | {note}")

    # ---------- KV cache 各精度 ----------
    print("\nKV cache 每 token（Qwen3-0.6B，2×28×8×128）在不同精度：")
    for prec, b in [("bf16", 2), ("fp8/int8", 1)]:
        print(f"  {prec:>9}: {2*28*8*128*b/1024:.0f} KiB/token")
    print("\n要点：")
    print("  - 量化【权重】→ 省权重显存 + 给'大模型'的 decode 提速（bandwidth-bound 时按字节比线性）；")
    print("  - 量化【KV cache】→ 直接砍第 2 章公式里的'每元素字节'→ 同显存塞更多并发 / 更长上下文；")
    print("  - 量化【激活】(W8A8) → 用 int8 Tensor Core，连 compute-bound 的 prefill 也能提速。")


if __name__ == "__main__":
    main()
```
<!-- CODE:exercises/01-inference/06_quantization.py END -->

---

[^quant]: Quantization（量化）：用更少比特表示权重/激活/KV 以省显存、提速。详见[术语表](../../glossary.md#quantization)。
[^gptq]: GPTQ：一种训练后 weight-only 量化方法，逐层最小化量化误差。详见[术语表](../../glossary.md#gptq)。
[^awq]: AWQ (Activation-aware Weight Quantization)：按激活重要性保护关键权重通道的 weight-only 量化。详见[术语表](../../glossary.md#awq)。
