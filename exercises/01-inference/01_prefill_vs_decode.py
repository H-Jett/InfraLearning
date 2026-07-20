#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 1 章练习：亲眼看见 Prefill 与 Decode 两个阶段的不同。

我们不调用高层的 model.generate()，而是手动拆开推理过程，这样才能
把 prefill（处理整段 prompt）和 decode（逐 token 生成）分别计时。

结论预期：
  1) Prefill 的“每 token 耗时”随 prompt 变长而暴跌（并行度高）；总耗时在小规模时
     被固定开销主导（近似平线），只有 token 足够多时才随长度明显上升。
  2) Decode 每个 token 的时间基本恒定（每步只算 1 个 token），单个 token 却可能比
     prefill 几千个 token 还慢 —— 因为 decode 是访存受限的（第 1 章正文详解）。

绝对路径：
  /volume/data/hjiang02/workspace/infra-learning/exercises/01-inference/01_prefill_vs_decode.py
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/volume/data/models/Qwen3-0.6B"
DEVICE = "cuda:0"


def sync():
    """CUDA 是异步的：kernel 发射后 CPU 立刻返回。计时前必须同步，否则测的是发射时间而非真实耗时。"""
    torch.cuda.synchronize()


def timeit(fn, warmup=1, repeat=3):
    """跑 warmup 次预热（第一次有 kernel 编译/缓存开销），再取 repeat 次的中位数。"""
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(repeat):
        sync(); t0 = time.perf_counter()
        fn()
        sync(); ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    print(f"加载模型: {MODEL_PATH}")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16
    ).to(DEVICE).eval()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.0f}M\n")

    # ---------- 实验 A：Prefill 时间随 prompt 长度变化 ----------
    print("=" * 60)
    print("实验 A：Prefill（处理整段 prompt 的一次前向）耗时 vs prompt 长度")
    print("=" * 60)
    print(f"{'prompt_len':>12} | {'prefill_ms':>12} | {'ms/token':>10}")
    print("-" * 40)
    for n in [16, 64, 256, 1024, 4096]:
        ids = torch.randint(0, tok.vocab_size, (1, n), device=DEVICE)

        @torch.no_grad()
        def prefill():
            return model(ids, use_cache=True)

        ms = timeit(prefill) * 1000
        print(f"{n:>12} | {ms:>12.2f} | {ms/n:>10.3f}")

    # ---------- 实验 B：Decode 每 token 耗时（用 KV cache 逐步生成） ----------
    print("\n" + "=" * 60)
    print("实验 B：Decode（借助 KV cache，每步只喂 1 个 token）每 token 耗时")
    print("=" * 60)
    prompt = "请用一句话解释什么是大语言模型的推理。"
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to(DEVICE)
    print(f"prompt token 数: {ids.shape[1]}")

    with torch.no_grad():
        # 第 1 步：prefill 整段 prompt，拿到第一个 token —— 这段时间就是 TTFT
        sync(); t0 = time.perf_counter()
        out = model(ids, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1:].argmax(-1)
        sync(); ttft = time.perf_counter() - t0
        print(f"TTFT（首 token 延迟, 含 prefill）: {ttft*1000:.2f} ms")

        # 后续步：每次只喂上一个 token + 复用 KV cache
        decode_times = []
        gen_ids = [next_id]
        for _ in range(50):
            sync(); t0 = time.perf_counter()
            out = model(next_id, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_id = out.logits[:, -1:].argmax(-1)
            sync(); decode_times.append(time.perf_counter() - t0)
            gen_ids.append(next_id)

    import statistics
    avg = statistics.mean(decode_times) * 1000
    print(f"TPOT（每 token 解码耗时, 50 步均值）: {avg:.2f} ms")
    print(f"解码吞吐: {1000/avg:.1f} tokens/s")
    gen = tok.decode(torch.cat(gen_ids, dim=-1)[0], skip_special_tokens=True)
    print(f"\n生成内容: {gen!r}")

    # ---------- 实验 C：KV cache 到底省了多少？关掉它对比 ----------
    print("\n" + "=" * 60)
    print("实验 C：关掉 KV cache 会怎样？（每步都重算整段历史）")
    print("=" * 60)
    with torch.no_grad():
        seq = ids.clone()
        # 有 cache：逐 token（上面已测，这里重测 20 步保持一致对比）
        past = model(ids, use_cache=True).past_key_values
        nid = ids[:, -1:]
        sync(); t0 = time.perf_counter()
        for _ in range(20):
            o = model(nid, past_key_values=past, use_cache=True)
            past = o.past_key_values
            nid = o.logits[:, -1:].argmax(-1)
        sync(); with_cache = time.perf_counter() - t0

        # 无 cache：每步把到目前为止的整段序列重新喂一遍
        seq = ids.clone()
        sync(); t0 = time.perf_counter()
        for _ in range(20):
            o = model(seq, use_cache=False)
            nid = o.logits[:, -1:].argmax(-1)
            seq = torch.cat([seq, nid], dim=-1)
        sync(); without_cache = time.perf_counter() - t0

    print(f"有 KV cache 生成 20 token: {with_cache*1000:.1f} ms")
    print(f"无 KV cache 生成 20 token: {without_cache*1000:.1f} ms")
    print(f"加速比: {without_cache/with_cache:.1f}x")


if __name__ == "__main__":
    main()
