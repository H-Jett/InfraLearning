#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 3 章练习：亲眼看见 batching 怎么把吞吐拉起来，以及"甜点 batch size"。

实验 A（真机）：固定生成长度，测不同 batch size 下的
    - 吞吐 throughput = batch × 生成token数 / 解码耗时
    - 每步延迟 per-step latency = 解码耗时 / 步数
    预期：batch 小时（memory-bound）吞吐随 batch 近乎线性涨、每步延迟几乎不变；
    batch 大到算力饱和（compute-bound）后，吞吐见顶、每步延迟开始明显上升。

实验 B（模拟）：说明 static batching 的"长尾浪费"——一批里生成长度不齐时，
    短序列早早结束却要陪跑到最长的那条，GPU 空转；continuous batching 能把这部分收回来。

绝对路径：
  /volume/data/hjiang02/workspace/infra-learning/exercises/01-inference/03_batching_throughput.py
"""

import time
import statistics
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/volume/data/models/Qwen3-0.6B"
DEVICE = "cuda:0"
GEN_TOKENS = 64


def sync():
    torch.cuda.synchronize()


@torch.no_grad()
def decode_time(model, input_ids, n_new):
    """prefill 一次，再逐 token decode n_new 步（batch 维度并行），返回纯 decode 耗时。"""
    out = model(input_ids, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1:].argmax(-1)
    sync()
    t0 = time.perf_counter()
    for _ in range(n_new):
        out = model(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
    sync()
    return time.perf_counter() - t0


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()

    prompt = "请简要介绍一下你自己。"
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    base = tok(text, return_tensors="pt").input_ids.to(DEVICE)

    # ---------- 实验 A：吞吐 vs batch size ----------
    print("=" * 70)
    print(f"实验 A：不同 batch size 下的吞吐与每步延迟（各生成 {GEN_TOKENS} 个 token）")
    print("=" * 70)
    print(f"{'batch':>6} | {'解码耗时(s)':>12} | {'每步延迟(ms)':>14} | {'吞吐(tok/s)':>14} | {'相对b=1加速':>12}")
    print("-" * 70)
    base_tp = None
    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        ids = base.repeat(bs, 1)
        # 预热一次
        decode_time(model, ids, 4)
        ts = [decode_time(model, ids, GEN_TOKENS) for _ in range(3)]
        t = statistics.median(ts)
        per_step_ms = t / GEN_TOKENS * 1000
        tp = bs * GEN_TOKENS / t
        if base_tp is None:
            base_tp = tp
        print(f"{bs:>6} | {t:>12.3f} | {per_step_ms:>14.2f} | {tp:>14.1f} | {tp/base_tp:>11.1f}x")

    print("\n读表：吞吐一路往上涨说明 batching 在生效（权重搬一次喂一批）；")
    print("      若某个 batch 之后吞吐涨幅骤减、每步延迟明显变大，那里就是 memory-bound → compute-bound 的翻转点。")

    # ---------- 实验 B：static batching 的长尾浪费（模拟） ----------
    print("\n" + "=" * 70)
    print("实验 B：static batching 的长尾浪费（模拟，不需要真跑模型）")
    print("=" * 70)
    # 造一批"生成长度不齐"的请求
    torch.manual_seed(0)
    lengths = torch.randint(8, 256, (32,)).tolist()  # 32 条请求，输出长度 8~255 不等
    n = len(lengths)
    longest = max(lengths)
    useful = sum(lengths)                 # 真正有用的 token-步
    static_cost = n * longest             # static：所有槽位陪跑到最长那条结束
    print(f"请求数 = {n}，输出长度范围 = [{min(lengths)}, {longest}]，总有用 token = {useful}")
    print(f"static batching 计算的 token-步 = {n} × {longest} = {static_cost}")
    print(f"其中浪费（短序列空转）= {static_cost - useful}  →  浪费率 {100*(1-useful/static_cost):.1f}%")
    print(f"continuous batching：完成即退出、空位即补新请求，基本只算 {useful} 个有用 token-步")
    print(f"  → 同样算力下，continuous batching 的有效吞吐约为 static 的 {static_cost/useful:.1f}x")


if __name__ == "__main__":
    main()
