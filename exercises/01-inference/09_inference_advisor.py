#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 9 章练习 / 第一部分总复盘：推理优化决策助手。

把前 8 课的规则写成代码：输入模型配置 + workload，自动估算显存/瓶颈，并给出
"该上哪些优化、为什么"的建议。不需要 GPU——这是一个把整部分知识串起来的决策工具。

用到的规则来自：
  第1章 memory/compute-bound、TPOT下限=模型字节/带宽
  第2章 KV_bytes = 2·层·KV头·head_dim·seqlen·batch·字节
  第3章 batching（高并发提吞吐）
  第4章 PagedAttention（显存碎片/并发）
  第5章 前缀复用（长共享前缀）
  第6章 量化（省显存/大模型decode提速）
  第7章 投机解码（低并发有效、高并发失效）

绝对路径：
  /volume/data/hjiang02/workspace/infra-learning/exercises/01-inference/09_inference_advisor.py
"""

from dataclasses import dataclass


@dataclass
class Model:
    name: str
    params: float          # 参数量
    layers: int
    kv_heads: int
    head_dim: int
    bytes_per: int = 2     # bf16


@dataclass
class Workload:
    name: str
    prompt_len: int
    gen_len: int
    batch: int             # 并发
    shared_prefix_frac: float  # 共享前缀占 prompt 比例 0~1
    latency_sensitive: bool
    gpu_mem_gb: float
    gpu_bw_tbps: float = 1.8


def gib(x): return x / 2**30


def advise(m: Model, w: Workload):
    print("=" * 66)
    print(f"模型 {m.name}（{m.params/1e9:.0f}B）  场景「{w.name}」")
    print("=" * 66)

    # --- 显存 ---
    model_bytes = m.params * m.bytes_per
    kv_per_tok = 2 * m.layers * m.kv_heads * m.head_dim * m.bytes_per
    seqlen = w.prompt_len + w.gen_len
    kv_total = kv_per_tok * seqlen * w.batch
    print(f"权重显存        ≈ {gib(model_bytes):.1f} GiB")
    print(f"KV cache/‌token  ≈ {kv_per_tok/1024:.0f} KiB")
    print(f"KV cache 总量   ≈ {gib(kv_total):.1f} GiB（seqlen {seqlen} × batch {w.batch}）")
    total_need = gib(model_bytes) + gib(kv_total)
    print(f"合计需求        ≈ {total_need:.1f} GiB / 可用 {w.gpu_mem_gb} GiB "
          f"{'⚠️ 超了！' if total_need > w.gpu_mem_gb else '✅'}")

    # --- 瓶颈判断 ---
    tpot_floor_ms = model_bytes / (w.gpu_bw_tbps * 1e12) * 1000
    print(f"\nTPOT 下限（模型字节/带宽）≈ {tpot_floor_ms:.1f} ms/token")
    decode_regime = "compute-bound（算力已被喂饱）" if w.batch >= 256 else "memory-bound（算力有富余）"
    print(f"decode 大致处于：{decode_regime}")

    # --- 建议 ---
    print("\n建议优化（按前 8 课规则）：")
    recs = []
    # PagedAttention + continuous batching：几乎总是要（引擎默认）
    recs.append(("PagedAttention + continuous batching", "引擎地基：消碎片、吞吐第一杠杆，用任何现代引擎都自带"))
    # 显存吃紧
    if total_need > w.gpu_mem_gb:
        recs.append(("KV cache 量化 / GQA / MLA", "显存超了，KV cache 是大头（第2章），砍它最直接"))
        recs.append(("权重量化 (int8/int4)", "腾出权重显存给 KV cache（第6章）"))
    # 长共享前缀
    if w.shared_prefix_frac >= 0.3:
        recs.append(("前缀复用 (Prefix Caching / RadixAttention)",
                     f"共享前缀占 {w.shared_prefix_frac*100:.0f}%，且 prompt 长，省大量重复 prefill（第5章）"))
    # 大模型 decode 提速
    if m.params >= 30e9 and tpot_floor_ms > 10:
        recs.append(("weight-only 量化 (W4A16)", "大模型 decode 是 bandwidth-bound，压权重≈按字节比提速（第6章）"))
    # 投机解码：低并发才有效
    if w.batch <= 8 and w.latency_sensitive:
        recs.append(("投机解码 (Speculative / EAGLE)", "低并发、算力有富余，无损降 TPOT（第7章）"))
    else:
        print("  ✗ 投机解码：高并发下目标已 compute-bound，收益小，跳过（第7章）")
    # 前缀复用不适用
    if w.shared_prefix_frac < 0.1:
        print("  ✗ 前缀复用：无共享前缀，用不上（第5章）")

    for i, (name, why) in enumerate(recs, 1):
        print(f"  {i}. {name}\n     → {why}")


if __name__ == "__main__":
    qwen70 = Model("Qwen-like-70B", 70e9, 80, 8, 128)
    # 场景 A：单用户长 system prompt 客服（低并发、延迟敏感、长共享前缀）
    advise(qwen70, Workload("单用户客服(长system prompt)", prompt_len=4000, gen_len=500,
                            batch=1, shared_prefix_frac=0.9, latency_sensitive=True, gpu_mem_gb=80))
    print()
    # 场景 B：离线批量改写（高吞吐、不在乎延迟、prompt 各不相同）
    advise(qwen70, Workload("离线批量处理", prompt_len=1000, gen_len=1000,
                            batch=256, shared_prefix_frac=0.0, latency_sensitive=False, gpu_mem_gb=80))
