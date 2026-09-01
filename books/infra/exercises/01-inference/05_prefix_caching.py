#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 5 章练习：量一量"前缀复用"能省多少 prefill / 把 TTFT 降多少。

场景：很多请求共享同一段长前缀（比如同一个长 system prompt、多轮对话的历史），
后面各接一小段不同的 query。

  - 朴素：每个请求都对"前缀 + query"整体做一次 prefill（重复算了前缀）。
  - 前缀复用：前缀的 KV 只算一次并缓存；每个请求只对自己那一小段 query 做 prefill，
    直接在缓存的前缀 KV 上接着算。

我们测每个请求的 TTFT（首 token 延迟 ≈ prefill 时间）在两种方式下的差别。
"""

import time
import statistics
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("INFRA_MODEL", "Qwen/Qwen3-0.6B")
DEVICE = "cuda:0"
QUERY_LEN = 16  # 每个请求各自的一小段 query 长度


def sync():
    torch.cuda.synchronize()


@torch.no_grad()
def ttft_naive(model, full_ids):
    """朴素：对 前缀+query 整体做一次 prefill。"""
    sync(); t0 = time.perf_counter()
    model(full_ids, use_cache=True)
    sync(); return time.perf_counter() - t0


@torch.no_grad()
def ttft_cached(model, prefix_ids, query_ids):
    """前缀复用：前缀 KV 已缓存（不计时），只对 query 段做 prefill。"""
    prefix_past = model(prefix_ids, use_cache=True).past_key_values  # 不计时：一次性、被所有请求摊薄
    sync(); t0 = time.perf_counter()
    model(query_ids, past_key_values=prefix_past, use_cache=True)
    sync(); return time.perf_counter() - t0


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()

    print(f"每个请求各自的 query 长度 = {QUERY_LEN} token\n")
    print(f"{'前缀长度':>8} | {'朴素 TTFT(ms)':>14} | {'前缀复用 TTFT(ms)':>18} | {'加速':>8} | {'省下 prefill':>12}")
    print("-" * 74)

    for prefix_len in [256, 1024, 4096, 16384]:
        prefix_ids = torch.randint(0, tok.vocab_size, (1, prefix_len), device=DEVICE)
        query_ids = torch.randint(0, tok.vocab_size, (1, QUERY_LEN), device=DEVICE)
        full_ids = torch.cat([prefix_ids, query_ids], dim=1)

        # 预热
        ttft_naive(model, full_ids); ttft_cached(model, prefix_ids, query_ids)

        naive = statistics.median([ttft_naive(model, full_ids) for _ in range(5)]) * 1000
        cached = statistics.median([ttft_cached(model, prefix_ids, query_ids) for _ in range(5)]) * 1000
        print(f"{prefix_len:>8} | {naive:>14.2f} | {cached:>18.2f} | {naive/cached:>7.1f}x | "
              f"{100*(1-cached/naive):>10.1f}%")

    print("\n结论：共享前缀越长，前缀复用省得越多——因为朴素每次都在重算这段前缀的 prefill，")
    print("      而前缀复用把它摊成'只算一次'。多轮对话、长 system prompt、few-shot 场景收益巨大。")


if __name__ == "__main__":
    main()
