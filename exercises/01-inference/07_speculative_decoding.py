#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 7 章练习：理解投机解码 (speculative decoding) 为什么能加速。

核心 enabler：大模型"一次并行验证 K 个 token"几乎和"解码 1 个 token"一样便宜——
因为 decode 是访存/发射受限，一次前向的成本主要是把权重搬一遍，多验几个 token 几乎免费
（就像 batching，只不过这次是沿"序列长度"方向）。

实验 A（真机）：给定一段前缀缓存，测目标模型一次前向处理 query_len = 1,2,4,8,16 个 token 的耗时。
    预期：基本是平线——验证 8 个 token ≈ 验证 1 个。这正是投机解码"免费验证"的基础。

实验 B（模拟）：给定接受率 α 和草稿长度 K，算"每次目标前向平均吐出多少 token"，
    从而估算理想加速比。

绝对路径：
  /volume/data/hjiang02/workspace/infra-learning/exercises/01-inference/07_speculative_decoding.py
"""

import time
import statistics
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/volume/data/models/Qwen3-0.6B"
DEVICE = "cuda:0"


def sync():
    torch.cuda.synchronize()


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()

    # ---------- 实验 A：一次验证 K 个 token 的耗时 ----------
    print("=" * 64)
    print("实验 A（真机）：目标模型一次前向处理 K 个 token 的耗时")
    print("=" * 64)
    prefix = torch.randint(0, tok.vocab_size, (1, 512), device=DEVICE)

    @torch.no_grad()
    def verify_time(k):
        past = model(prefix, use_cache=True).past_key_values  # 前缀缓存，不计时
        q = torch.randint(0, tok.vocab_size, (1, k), device=DEVICE)
        sync(); t0 = time.perf_counter()
        model(q, past_key_values=past, use_cache=True)
        sync(); return time.perf_counter() - t0

    with torch.no_grad():
        verify_time(1)  # 预热
        print(f"{'K (验证token数)':>16} | {'一次前向(ms)':>14} | {'摊到每token(ms)':>16}")
        print("-" * 54)
        for k in [1, 2, 4, 8, 16]:
            ms = statistics.median([verify_time(k) for _ in range(5)]) * 1000
            print(f"{k:>16} | {ms:>14.2f} | {ms/k:>16.2f}")
    print("\n看：一次前向的总耗时几乎不随 K 变（平线）——验证 8 个 token ≈ 验证 1 个。")
    print("这就是投机解码的基础：既然多验几个几乎免费，那就'先猜后验'，一次验证收获多个 token。")

    # ---------- 实验 B：接受率 → 每次目标前向吐出的 token 数 ----------
    print("\n" + "=" * 64)
    print("实验 B（模拟）：接受率 α、草稿长度 K → 每次目标前向平均吐出多少 token")
    print("=" * 64)
    print("公式：E[tokens] = (1 - α^(K+1)) / (1 - α)   （含 1 个 bonus token）")
    print("标准解码每次目标前向只吐 1 个 token；投机解码吐 E 个 → 理想加速 ≈ E 倍（未扣草稿开销）\n")
    print(f"{'接受率 α':>10} | " + " | ".join(f"K={k:<2}" for k in [2, 4, 8]))
    print("-" * 44)
    for alpha in [0.3, 0.5, 0.7, 0.8, 0.9]:
        row = []
        for K in [2, 4, 8]:
            E = (1 - alpha ** (K + 1)) / (1 - alpha)
            row.append(f"{E:>4.2f}")
        print(f"{alpha:>10.1f} | " + " |  ".join(row))
    print("\n要点：")
    print("  - 接受率 α 越高（草稿越像目标），每次验证吐出的 token 越多，加速越大；")
    print("  - α 太低时几乎不加速，甚至因草稿开销变慢；")
    print("  - 关键：无论接受多少，最终分布严格等于目标模型 → 投机解码是【无损】的（不掉精度）。")
    print("  - 注意：高 batch 时目标模型已 compute-bound，'免费验证'不再免费，投机解码收益变小")
    print("    （所以它主要用在低延迟、低并发场景）。")


if __name__ == "__main__":
    main()
