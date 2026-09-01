#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成第 1 章的配图（输出到 docs/chapters/01-vision-basics/figures/）：

  fig1  ch1_patchify.png        —— 一张图被切成 patch 网格，以及单个 patch 放大后的样子
  fig2  ch1_tokens_vs_pixels.png —— Qwen2.5-VL 实测：视觉 token 数随像素线性增长

配色遵循 Okabe-Ito 色盲安全方案，白底、英文标签。
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import sample_image  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_DIR = os.path.join(REPO_ROOT, "docs", "chapters", "01-vision-basics", "figures")
BLUE, ORANGE, VERMILLION, GREEN = "#0072B2", "#E69F00", "#D55E00", "#009E73"

VLM_MODEL = os.environ.get("MM_VLM", "Qwen/Qwen2.5-VL-3B-Instruct")


def fig_patchify(img):
    """左：224×224 原图 + 16×16 patch 网格；中：放大的单个 patch；右：该 patch 的像素值。"""
    P, RES = 16, 224
    small = img.resize((RES, RES))
    arr = np.asarray(small)
    grid = RES // P

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    ax = axes[0]
    ax.imshow(arr)
    for i in range(grid + 1):
        ax.axhline(i * P - 0.5, color="white", lw=0.6, alpha=0.8)
        ax.axvline(i * P - 0.5, color="white", lw=0.6, alpha=0.8)
    r, c = 6, 7  # 高亮其中一个 patch
    ax.add_patch(plt.Rectangle((c * P - 0.5, r * P - 0.5), P, P,
                               fill=False, edgecolor=ORANGE, lw=2.5))
    ax.set_title(f"224x224 image -> {grid}x{grid} = {grid * grid} patches", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    patch = arr[r * P:(r + 1) * P, c * P:(c + 1) * P]
    ax.imshow(patch, interpolation="nearest")
    ax.set_title(f"one 16x16 patch\n(= {P * P * 3} numbers)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(ORANGE); spine.set_linewidth(2.5)

    ax = axes[2]
    flat = patch.reshape(-1, 3).mean(axis=1) / 255.0
    ax.plot(flat, color=BLUE, lw=0.9)
    ax.set_title("flattened -> a vector, then Linear -> d_model\n"
                 "(plotted: mean over RGB, so 256 of the 768 values)", fontsize=10)
    ax.set_xlabel("position in patch"); ax.set_ylabel("pixel value (mean of RGB)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "ch1_patchify.png")
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[图] {os.path.relpath(out, REPO_ROOT)}")


def fig_tokens_vs_pixels(img):
    """实测 Qwen2.5-VL 的视觉 token 数 vs 像素数，并和 224² 的 ViT 基准对比。"""
    from transformers import AutoProcessor

    ip = AutoProcessor.from_pretrained(VLM_MODEL).image_processor
    merge2 = ip.merge_size ** 2

    sizes = [(224, 224), (336, 336), (448, 448), (640, 480), (960, 686),
             (1280, 960), (1600, 1200), (1920, 1080)]
    px, toks, labels = [], [], []
    for w, h in sizes:
        enc = ip(images=img.resize((w, h)), return_tensors="pt")
        t, gh, gw = enc["image_grid_thw"][0].tolist()
        toks.append(t * gh * gw // merge2)
        px.append(w * h / 1e6)
        labels.append(f"{w}x{h}")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(px, toks, "o-", color=BLUE, lw=1.6, ms=5, label="Qwen2.5-VL (native res, 2x2 merge)")

    # 参考线：固定 224×224 输入的 ViT-B/16，无论多大的图都只有 196 个 token
    ax.axhline(196, color=ORANGE, ls="--", lw=1.4,
               label="ViT-B/16 @ fixed 224x224 = 196 tokens")

    for x, y, lb in zip(px, toks, labels):
        if lb in ("224x224", "640x480", "960x686", "1920x1080", "1600x1200"):
            ax.annotate(f"{lb}\n{y} tok", (x, y), textcoords="offset points",
                        xytext=(6, -14), fontsize=8, color="#333333")

    ax.set_xlabel("image size (megapixels)")
    ax.set_ylabel("vision tokens fed into the LLM")
    ax.set_title("Vision tokens grow linearly with pixel count", fontsize=12)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "ch1_tokens_vs_pixels.png")
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[图] {os.path.relpath(out, REPO_ROOT)}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    img = sample_image()
    fig_patchify(img)
    fig_tokens_vs_pixels(img)


if __name__ == "__main__":
    main()
