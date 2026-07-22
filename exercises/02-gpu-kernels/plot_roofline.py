#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成第 10 章的 Roofline 示意图（用 RTX 5090 实测数据）。
输出：docs/chapters/02-gpu-kernels/figures/roofline.png

设计遵循数据可视化规范：
  - 形式：log-log 折线（roofline 天然是分段边界）；
  - 配色：Okabe-Ito 色盲安全调色板；
  - 细线 + 淡网格 + 直接标注，白底（明暗模式都可读）；
  - 图内文字用英文，避免 matplotlib 缺中文字体显示方框。

绝对路径：
  /volume/data/hjiang02/workspace/infra-learning/exercises/02-gpu-kernels/plot_roofline.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 实测数据（第 10 章练习）---
PEAK_TFLOPS = 236.7      # bf16 峰值算力
BW_TBPS = 1.570          # 显存带宽 TB/s
RIDGE = PEAK_TFLOPS / BW_TBPS   # 拐点算术强度 ≈ 151

# Okabe-Ito 色盲安全色
BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#8a8f98"

OUT = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "docs", "chapters", "02-gpu-kernels", "figures", "roofline.png",
)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ai = np.logspace(-1, 4, 500)                 # 算术强度 0.1 ~ 10000 FLOP/byte
    roof = np.minimum(PEAK_TFLOPS, BW_TBPS * ai)  # roofline 边界

    fig, ax = plt.subplots(figsize=(7, 4.3), dpi=130)

    # roofline 边界（主线）
    ax.plot(ai, roof, color="#222", lw=2, zorder=3, label="Roofline (attainable)")
    ax.fill_between(ai, roof, 0.1, color="#222", alpha=0.05, zorder=0)

    # 拐点
    ax.scatter([RIDGE], [PEAK_TFLOPS], color="#222", s=30, zorder=4)
    ax.annotate(f"ridge ≈ {RIDGE:.0f} FLOP/byte",
                xy=(RIDGE, PEAK_TFLOPS), xytext=(RIDGE * 1.3, PEAK_TFLOPS * 0.42),
                fontsize=9, color="#222",
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1))

    # 两个示例工作负载
    ax.scatter([1], [BW_TBPS * 1], color=BLUE, s=55, zorder=5)
    ax.annotate("decode\n(AI≈1, memory-bound)", xy=(1, BW_TBPS),
                xytext=(1.3, BW_TBPS * 2.6), fontsize=9, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))
    ax.scatter([500], [PEAK_TFLOPS], color=ORANGE, s=55, zorder=5)
    ax.annotate("large matmul\n(compute-bound)", xy=(500, PEAK_TFLOPS),
                xytext=(500, PEAK_TFLOPS * 0.28), fontsize=9, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1))

    # 屋顶标注
    ax.text(2000, PEAK_TFLOPS * 1.06, f"peak compute {PEAK_TFLOPS:.0f} TFLOPS (bf16)",
            fontsize=8.5, color=GRAY, ha="right")
    ax.text(0.5, 3.0, f"bandwidth roof\n{BW_TBPS*1000:.0f} GB/s × AI",
            fontsize=8.5, color=GRAY, rotation=32)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)")
    ax.set_ylabel("Attainable (TFLOPS)")
    ax.set_title("Roofline — RTX 5090 (measured)", fontsize=11, fontweight="bold")
    ax.set_xlim(0.1, 1e4); ax.set_ylim(0.5, PEAK_TFLOPS * 2)
    ax.grid(True, which="both", color=GRAY, alpha=0.18, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print("saved:", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
