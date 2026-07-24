#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 4 章练习：用模拟对比"预留最大长度"和"分页 (PagedAttention)"两种 KV cache 显存管理，
看后者能在同样显存里塞下多少倍的并发序列、显存利用率差多少。

不需要 GPU：这是一个显存分配的模拟，用来把 PagedAttention 的收益量化出来。

两种策略：
  1) 朴素·预留最大长度：每条活跃序列都按模型最大上下文预留一整块连续显存
     （不管它实际会生成多长）→ 巨大的内部浪费。
  2) 分页 (PagedAttention)：把 KV cache 切成固定大小的 block，按需一块块分配
     → 只在每条序列的"最后一块"有一点内部碎片，利用率极高。
"""

import math
import random

# --- 用第 2 章的 Qwen3-0.6B 真实参数算"每 token 的 KV 字节" ---
LAYERS, KV_HEADS, HEAD_DIM, BYTES = 28, 8, 128, 2
PER_TOKEN_BYTES = 2 * LAYERS * KV_HEADS * HEAD_DIM * BYTES  # = 114688 B = 112 KiB

MAX_LEN = 40960          # 模型最大上下文
BLOCK = 16               # PagedAttention 每个 block 的 token 数（vLLM 默认 16）
KV_BUDGET = 8 * 2**30    # 假设留给 KV cache 的显存预算：8 GiB


def human(n):
    for u in ["B", "KiB", "MiB", "GiB"]:
        if n < 1024 or u == "GiB":
            return f"{n:.2f} {u}"
        n /= 1024


def main():
    print(f"每 token KV = {human(PER_TOKEN_BYTES)}，KV 显存预算 = {human(KV_BUDGET)}，"
          f"模型最大上下文 = {MAX_LEN}，block = {BLOCK} tokens\n")

    # 造一批真实的"长度不齐"的请求：大多数短、少数长
    random.seed(0)
    seqs = [min(MAX_LEN, int(random.expovariate(1 / 400)) + 8) for _ in range(2000)]
    avg = sum(seqs) / len(seqs)
    print(f"请求样本 {len(seqs)} 条，平均实际长度 = {avg:.0f} tokens，最长 = {max(seqs)}\n")

    # ---------- 策略 1：预留最大长度 ----------
    reserve_per_seq = MAX_LEN * PER_TOKEN_BYTES
    n_naive = KV_BUDGET // reserve_per_seq
    used_if_naive = n_naive * avg * PER_TOKEN_BYTES
    print("=" * 60)
    print("策略 1：朴素·每条序列预留最大长度的连续显存")
    print("=" * 60)
    print(f"每条预留 = {MAX_LEN} × {human(PER_TOKEN_BYTES)} = {human(reserve_per_seq)}")
    print(f"能容纳并发序列数 = {n_naive}")
    print(f"显存利用率 ≈ 实际用量/预留 = 平均长度/最大长度 = {avg/MAX_LEN*100:.2f}%")

    # ---------- 策略 2：分页 ----------
    total_blocks = KV_BUDGET // (BLOCK * PER_TOKEN_BYTES)
    blocks_per_seq = [math.ceil(s / BLOCK) for s in seqs]
    avg_blocks = sum(blocks_per_seq) / len(blocks_per_seq)
    n_paged = total_blocks / avg_blocks
    # 内部碎片：每条序列最后一个 block 平均浪费 (BLOCK-1)/2 个 token 的空间
    used_tokens = sum(seqs)
    alloc_tokens = sum(blocks_per_seq) * BLOCK
    print("\n" + "=" * 60)
    print("策略 2：分页 PagedAttention（按需分配 block）")
    print("=" * 60)
    print(f"显存池总 block 数 = {total_blocks}")
    print(f"每条序列平均占用 = {avg_blocks:.1f} 个 block（{human(avg_blocks*BLOCK*PER_TOKEN_BYTES)}）")
    print(f"能容纳并发序列数 ≈ {n_paged:.0f}")
    print(f"显存利用率 ≈ 有效token/已分配token = {used_tokens/alloc_tokens*100:.2f}%（只剩最后一块的内部碎片）")

    # ---------- 对比 ----------
    print("\n" + "=" * 60)
    print("对比")
    print("=" * 60)
    print(f"并发能力：分页 / 朴素 ≈ {n_paged/n_naive:.0f}x")
    print(f"利用率  ：{avg/MAX_LEN*100:.1f}%  →  {used_tokens/alloc_tokens*100:.1f}%")
    print("\n结论：朴素法为了'万一序列很长'给每条都预留最大长度，绝大部分显存被闲置；")
    print("      分页只按实际长度一块块给，利用率逼近 100%，同样显存能多塞几十倍并发。")
    print("      这就是 vLLM 用 PagedAttention 把吞吐拉高的根本原因。")


if __name__ == "__main__":
    main()
