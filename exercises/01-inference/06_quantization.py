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
