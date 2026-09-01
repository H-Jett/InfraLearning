#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 8 章练习：把"快"量化成服务指标，画出延迟-吞吐曲线，并理解 goodput。

概念：
  - TTFT (Time To First Token)：首 token 延迟，主要由 prefill 决定。
  - TPOT (Time Per Output Token)：解码阶段每个后续 token 的间隔。
  - 吞吐 (throughput)：整个系统每秒产出多少 token。
  - SLO (Service Level Objective)：延迟目标，如 "TPOT ≤ 25ms"。
  - goodput (有效吞吐)：只有满足 SLO 的吞吐才算数——越过拐点后吞吐虽高但违反 SLO，不算 goodput。

实验：扫不同并发 batch，测 TTFT / TPOT / 吞吐，按 SLO 算 goodput，找到"最大 goodput"的甜点。
"""

import time
import statistics
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("INFRA_MODEL", "Qwen/Qwen3-0.6B")
DEVICE = "cuda:0"
GEN = 64
SLO_TPOT_MS = 25.0  # 服务级目标：每 token 延迟不超过 25ms


def sync():
    torch.cuda.synchronize()


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()
    text = tok.apply_chat_template(
        [{"role": "user", "content": "请简要介绍一下你自己。"}],
        tokenize=False, add_generation_prompt=True,
    )
    base = tok(text, return_tensors="pt").input_ids.to(DEVICE)

    @torch.no_grad()
    def measure(bs):
        ids = base.repeat(bs, 1)
        # TTFT：prefill 一次的时间
        sync(); t0 = time.perf_counter()
        out = model(ids, use_cache=True)
        sync(); ttft = time.perf_counter() - t0
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
        # TPOT：decode 每步
        sync(); t0 = time.perf_counter()
        for _ in range(GEN):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
        sync(); tpot = (time.perf_counter() - t0) / GEN
        return ttft * 1000, tpot * 1000, bs / tpot  # ms, ms, tok/s(system)

    print(f"SLO：TPOT ≤ {SLO_TPOT_MS} ms\n")
    print(f"{'并发':>6} | {'TTFT(ms)':>9} | {'TPOT(ms)':>9} | {'吞吐(tok/s)':>12} | {'满足SLO':>7} | {'goodput':>10}")
    print("-" * 66)
    best = (0, 0)
    for bs in [1, 8, 32, 64, 128, 256, 512, 1024]:
        measure(min(bs, 4))  # 预热
        ttft, tpot, tp = min((measure(bs) for _ in range(3)), key=lambda r: r[1])
        ok = tpot <= SLO_TPOT_MS
        good = tp if ok else 0.0
        if good > best[1]:
            best = (bs, good)
        print(f"{bs:>6} | {ttft:>9.1f} | {tpot:>9.2f} | {tp:>12.0f} | {'✅' if ok else '❌':>6} | {good:>10.0f}")

    print(f"\n最大 goodput 出现在并发 = {best[0]}（{best[1]:.0f} tok/s）——再往上虽然原始吞吐还涨，")
    print("但 TPOT 越过 SLO，那些请求'超时'不算有效服务，goodput 掉回 0。")
    print("要点：调服务不是最大化吞吐，而是【在满足 SLO 的前提下】最大化 goodput。")


if __name__ == "__main__":
    main()
