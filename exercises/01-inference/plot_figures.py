#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成第一部分各章的数据图（用会话中真机实测的数字重画，不重跑模型）。
输出到 docs/chapters/01-inference/figures/。

规范：Okabe-Ito 色盲安全配色、白底、英文标签（避免 matplotlib 缺中文字体出方框）、
细线 + 淡网格、直接标注。

绝对路径：
  /volume/data/hjiang02/workspace/infra-learning/exercises/01-inference/plot_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, VERM, GREEN, GRAY = "#0072B2", "#E69F00", "#D55E00", "#009E73", "#8a8f98"
FIG = os.path.join(os.path.dirname(__file__), "..", "..",
                   "docs", "chapters", "01-inference", "figures")


def _style(ax):
    ax.grid(True, which="both", color=GRAY, alpha=0.18, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", os.path.normpath(p))


# ---- 第1章：decode 每 token 比 prefill 贵多少（ms/token，log y）----
def ch1():
    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=130)
    labels = ["prefill\n(4096 tok)", "decode\n(1 tok)"]
    vals = [0.017, 37.9]  # ms/token
    bars = ax.bar(labels, vals, color=[GREEN, VERM], width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("time per token (ms, log)")
    ax.set_title("Decode is ~2000× more expensive per token than prefill", fontsize=10, fontweight="bold")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:g} ms", ha="center", fontsize=9)
    _style(ax)
    save(fig, "ch1_prefill_vs_decode.png")


# ---- 第2章：KV cache 随 seq_len 增长，与模型权重交叉 ----
def ch2():
    per_tok = 114688  # bytes
    seq = np.linspace(1, 40960, 400)
    kv_gib = per_tok * seq / 2**30
    model_gib = 1.11
    cross = model_gib * 2**30 / per_tok
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=130)
    ax.plot(seq, kv_gib, color=BLUE, lw=2, label="KV cache (batch=1)")
    ax.axhline(model_gib, color=ORANGE, lw=2, ls="--", label="model weights (0.6B, bf16)")
    ax.axvline(cross, color=GRAY, lw=1, ls=":")
    ax.annotate(f"crossover ≈ {cross:.0f} tokens\n(KV cache = model weights)",
                xy=(cross, model_gib), xytext=(cross * 1.15, model_gib * 2.4),
                fontsize=9, color="#222", arrowprops=dict(arrowstyle="->", color=GRAY, lw=1))
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("memory (GiB)")
    ax.set_title("KV cache grows linearly and overtakes the model itself", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    save(fig, "ch2_kv_cache_growth.png")


# ---- 第3章：吞吐 vs batch，拐点 512→1024 ----
def ch3():
    batch = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    tp = [58, 115, 229, 458, 923, 1817, 3644, 7334, 14186, 27080, 34117]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=130)
    ax.plot(batch, tp, color=BLUE, lw=2, marker="o", ms=4)
    ax.scatter([512], [27080], color=GREEN, s=70, zorder=5)
    ax.annotate("sweet spot 512\n(throughput up, latency flat)", xy=(512, 27080),
                xytext=(20, 30000), fontsize=9, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))
    ax.scatter([1024], [34117], color=VERM, s=70, zorder=5)
    ax.annotate("1024: knee\n(+26% tp, +60% latency)", xy=(1024, 34117),
                xytext=(120, 6000), fontsize=9, color=VERM,
                arrowprops=dict(arrowstyle="->", color=VERM, lw=1))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("batch (concurrency)")
    ax.set_ylabel("throughput (tok/s)")
    ax.set_title("Batching: throughput scales ~linearly, then knees", fontsize=10, fontweight="bold")
    _style(ax)
    save(fig, "ch3_batching_throughput.png")


# ---- 第5章：前缀复用 TTFT（短前缀无用，长前缀大省）----
def ch5():
    plen = [256, 1024, 4096, 16384]
    naive = [19.3, 19.5, 55.7, 335.5]
    cached = [20.3, 20.3, 20.5, 46.8]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=130)
    ax.plot(plen, naive, color=VERM, lw=2, marker="o", ms=5, label="naive (re-prefill)")
    ax.plot(plen, cached, color=BLUE, lw=2, marker="s", ms=5, label="prefix caching")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("shared prefix length (tokens)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Prefix caching: no gain when prefix is short, big win when long", fontsize=9.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    save(fig, "ch5_prefix_caching.png")


# ---- 第6章：权重显存 vs 精度 ----
def ch6():
    params = 596e6
    precs = ["fp32", "bf16", "int8", "int4"]
    bytes_per = [4, 2, 1, 0.5]
    gib = [params * b / 2**30 for b in bytes_per]
    fig, ax = plt.subplots(figsize=(5.8, 3.6), dpi=130)
    bars = ax.bar(precs, gib, color=[GRAY, BLUE, GREEN, ORANGE], width=0.62)
    for b, v in zip(bars, gib):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("weight memory (GiB)")
    ax.set_title("Quantization halves memory each step (0.6B model)", fontsize=10, fontweight="bold")
    _style(ax)
    save(fig, "ch6_quant_memory.png")


# ---- 第7章：接受率 → 每次目标前向吐出的 token 数 ----
def ch7():
    alpha = np.linspace(0.2, 0.95, 60)
    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=130)
    for K, c, lab in [(2, GREEN, "K=2"), (4, BLUE, "K=4"), (8, VERM, "K=8")]:
        E = (1 - alpha ** (K + 1)) / (1 - alpha)
        ax.plot(alpha, E, color=c, lw=2, label=lab)
    ax.axhline(1, color=GRAY, lw=1, ls=":")
    ax.text(0.21, 1.06, "baseline = 1 token / forward", color=GRAY, fontsize=8)
    ax.set_xlabel("acceptance rate α")
    ax.set_ylabel("tokens per target forward (≈ speedup)")
    ax.set_title("Speculative decoding: higher acceptance → more tokens per verify", fontsize=9.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    save(fig, "ch7_speculative.png")


# ---- 第8章：延迟-吞吐 + SLO 线 + goodput ----
def ch8():
    tp = [59, 465, 1848, 3689, 7415, 14492, 27256, 34071]
    tpot = [16.91, 17.20, 17.31, 17.35, 17.26, 17.66, 18.79, 30.05]
    lab = ["1", "8", "32", "64", "128", "256", "512", "1024"]
    slo = 25.0
    fig, ax = plt.subplots(figsize=(6.6, 3.9), dpi=130)
    colors = [BLUE if t <= slo else VERM for t in tpot]
    ax.plot(tp, tpot, color=GRAY, lw=1.5, zorder=1)
    ax.scatter(tp, tpot, c=colors, s=55, zorder=3)
    for x, y, l in zip(tp, tpot, lab):
        ax.annotate(l, (x, y), textcoords="offset points", xytext=(4, 5), fontsize=8, color="#444")
    ax.axhline(slo, color=ORANGE, lw=1.6, ls="--")
    ax.text(60, slo + 0.6, f"SLO: TPOT ≤ {slo:.0f} ms", color=ORANGE, fontsize=9)
    ax.annotate("max goodput\n(512, meets SLO)", xy=(27256, 18.79), xytext=(6000, 21.5),
                fontsize=9, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))
    ax.annotate("1024: highest throughput\nbut goodput = 0 (SLO violated)", xy=(34071, 30.05),
                xytext=(3000, 27.5), fontsize=9, color=VERM,
                arrowprops=dict(arrowstyle="->", color=VERM, lw=1))
    ax.set_xscale("log")
    ax.set_xlabel("throughput (tok/s, log)")
    ax.set_ylabel("TPOT (ms/token)")
    ax.set_title("Latency–throughput: past the SLO, extra throughput is worthless", fontsize=9.5, fontweight="bold")
    _style(ax)
    save(fig, "ch8_serving_goodput.png")


if __name__ == "__main__":
    ch1(); ch2(); ch3(); ch5(); ch6(); ch7(); ch8()
    print("done.")
