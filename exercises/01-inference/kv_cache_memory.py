#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 2 章练习：把 KV Cache 的显存"一字节字节"算出来，并和真机实测对上。

我们要验证的公式（单位：字节）：
    KV_bytes = 2 × num_layers × num_kv_heads × head_dim × seq_len × batch × bytes_per_elem
其中 2 = K 和 V 各一份。注意是 num_kv_heads（GQA/MQA 会小于 attention heads）。

三件事：
  A) 用 Qwen3-0.6B 的真实 config 手算 per-token 和不同长度的 KV cache 大小；
  B) 真机跑一遍，量出 past_key_values 的实际字节数，和公式对比；
  C) 对比 GQA vs 若为 MHA 的差异，以及 KV cache 与模型权重的"交叉点"。

绝对路径：
  /volume/data/hjiang02/workspace/infra-learning/exercises/01-inference/kv_cache_memory.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/volume/data/models/Qwen3-0.6B"
DEVICE = "cuda:0"


def kv_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch, bytes_per_elem):
    """KV cache 显存公式。"""
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_elem


def human(n):
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if n < 1024 or unit == "GiB":
            return f"{n:.2f} {unit}"
        n /= 1024


def measure_cache_bytes(past):
    """把 past_key_values 里所有张量的字节数加总（兼容 DynamicCache 和旧式 tuple）。"""
    tensors = []
    if hasattr(past, "key_cache"):          # 较新的 DynamicCache
        tensors = list(past.key_cache) + list(past.value_cache)
    elif hasattr(past, "layers"):           # 更新版本按 layer 组织
        for layer in past.layers:
            tensors += [getattr(layer, "keys", None), getattr(layer, "values", None)]
    else:                                    # 旧式：tuple(tuple(K, V))
        for layer in past:
            tensors += list(layer)
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def main():
    cfg = AutoModelForCausalLM.from_pretrained(MODEL_PATH).config
    L = cfg.num_hidden_layers
    H_attn = cfg.num_attention_heads
    H_kv = cfg.num_key_value_heads
    d = cfg.head_dim
    b = 2  # bf16 = 2 字节
    print("=" * 64)
    print("Qwen3-0.6B 关键 config")
    print("=" * 64)
    print(f"层数 num_hidden_layers      = {L}")
    print(f"注意力头 num_attention_heads = {H_attn}")
    print(f"KV 头   num_key_value_heads = {H_kv}   ← GQA：比 Q 头少！")
    print(f"每头维度 head_dim            = {d}")
    print(f"精度                         = bf16 ({b} 字节)")

    # ---------- A) 手算 ----------
    print("\n" + "=" * 64)
    print("A) 用公式手算 KV cache 大小（batch=1）")
    print("=" * 64)
    per_tok = kv_bytes(L, H_kv, d, 1, 1, b)
    print(f"每个 token 的 KV cache = 2×{L}×{H_kv}×{d}×{b} = {per_tok} 字节 = {human(per_tok)}")
    print(f"\n{'seq_len':>10} | {'KV cache':>14}")
    print("-" * 28)
    for n in [128, 1024, 4096, 40960]:
        print(f"{n:>10} | {human(kv_bytes(L, H_kv, d, n, 1, b)):>14}")

    # ---------- B) 真机实测，和公式对比 ----------
    print("\n" + "=" * 64)
    print("B) 真机实测 past_key_values 字节数 vs 公式")
    print("=" * 64)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()

    print(f"{'seq_len':>10} | {'实测':>14} | {'公式':>14} | 是否一致")
    print("-" * 56)
    with torch.no_grad():
        for n in [128, 1024, 4096]:
            ids = torch.randint(0, tok.vocab_size, (1, n), device=DEVICE)
            out = model(ids, use_cache=True)
            measured = measure_cache_bytes(out.past_key_values)
            formula = kv_bytes(L, H_kv, d, n, 1, b)
            ok = "✅" if measured == formula else "❌"
            print(f"{n:>10} | {human(measured):>14} | {human(formula):>14} | {ok}")

    # ---------- C) GQA vs MHA，以及与模型权重的对比 ----------
    print("\n" + "=" * 64)
    print("C) GQA 省了多少？KV cache 何时超过模型权重？")
    print("=" * 64)
    n = cfg.max_position_embeddings
    kv_gqa = kv_bytes(L, H_kv, d, n, 1, b)
    kv_mha = kv_bytes(L, H_attn, d, n, 1, b)  # 假设不用 GQA，KV 头 = attention 头
    n_params = sum(p.numel() for p in model.parameters())
    model_bytes = n_params * 2  # bf16
    print(f"最大上下文 seq_len = {n}")
    print(f"KV cache（GQA, {H_kv} KV 头）      = {human(kv_gqa)}")
    print(f"KV cache（若 MHA, {H_attn} KV 头）  = {human(kv_mha)}   → GQA 省了 {kv_mha/kv_gqa:.1f}x")
    print(f"模型权重（{n_params/1e6:.0f}M 参数, bf16） = {human(model_bytes)}")
    print(f"\n结论：满上下文时单条序列的 KV cache（{human(kv_gqa)}）已 {kv_gqa/model_bytes:.1f}× 于模型权重本身！")
    print("这就是为什么长上下文 / 高并发下，KV cache 是头号显存杀手。")


if __name__ == "__main__":
    main()
