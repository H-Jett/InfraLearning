#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成第 3 章的配图（输出到 docs/chapters/01-vision-basics/figures/）：

  ch3_similarity_matrix.png —— 4 图 × 6 句的 cosine 相似度矩阵（对比学习要拉高的就是对角线）
  ch3_temperature.png       —— 温度 τ 怎么把相似度变成概率：最高概率与熵随 τ 变化

数字全部来自 CLIP ViT-B/32 真机前向。配色 Okabe-Ito，白底，英文标签。
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import sample_images  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_DIR = os.path.join(REPO_ROOT, "docs", "chapters", "01-vision-basics", "figures")
BLUE, ORANGE, VERMILLION = "#0072B2", "#E69F00", "#D55E00"

CLIP_MODEL = os.environ.get("MM_CLIP", "openai/clip-vit-base-patch32")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

TEXTS = [
    "wild cat in the snow",
    "two cats + remotes",
    "living room with TV",
    "stone house, lawn",
    "a bowl of noodles",
    "neural network diagram",
]
FULL_TEXTS = [
    "a close-up photo of a wild cat standing in the snow",
    "two tabby cats lying on a pink blanket with two remote controls",
    "an indoor living room with a television and a dining table",
    "an old stone house with a red tiled roof and a green lawn",
    "a bowl of noodles",
    "a diagram of a neural network",
]


def compute(imgs):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    with torch.no_grad():
        px = proc(images=[i for _, i, _ in imgs], return_tensors="pt").to(DEVICE)["pixel_values"]
        tk = proc(text=FULL_TEXTS, return_tensors="pt", padding=True).to(DEVICE)
        fi = model.get_image_features(pixel_values=px)
        ft = model.get_text_features(**tk)
    fi = fi / fi.norm(dim=-1, keepdim=True)
    ft = ft / ft.norm(dim=-1, keepdim=True)
    return (fi @ ft.T).cpu().numpy(), model.logit_scale.exp().item()


def fig_matrix(sim, names):
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    im = ax.imshow(sim, cmap="magma", vmin=sim.min(), vmax=sim.max())
    ax.set_xticks(range(len(TEXTS)))
    ax.set_xticklabels(TEXTS, rotation=28, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if sim[i, j] < sim.max() * 0.75 else "black")
    # 框出应该最亮的对角线（前 4 列与 4 张图一一对应）
    for i in range(min(sim.shape[0], 4)):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="#00E5A0", lw=2.2))
    ax.set_title("CLIP cosine similarity: training pushes the boxed diagonal up", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "ch3_similarity_matrix.png")
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[图] {os.path.relpath(out, REPO_ROOT)}")


def fig_temperature(sim, scale):
    row = torch.from_numpy(sim[1])              # 以「两只猫」那行为例
    taus = np.logspace(np.log10(0.003), np.log10(2.0), 60)
    maxp, ent = [], []
    for t in taus:
        p = (row / float(t)).softmax(dim=-1)
        maxp.append(p.max().item())
        ent.append(-(p * p.clamp_min(1e-12).log()).sum().item())

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, ys, lab, color in [(axes[0], maxp, "max probability", BLUE),
                               (axes[1], ent, "entropy (nats)", VERMILLION)]:
        ax.plot(taus, ys, color=color, lw=1.8)
        ax.axvline(1 / scale, color=ORANGE, ls="--", lw=1.4)
        ax.annotate(f"CLIP's learned\ntau = {1 / scale:.3f}", (1 / scale, max(ys) * 0.55),
                    fontsize=8.5, color="#555555", xytext=(6, 0), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("temperature tau (log scale)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_title("sharper as tau shrinks", fontsize=11)
    axes[1].set_title("entropy collapses to ~0", fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "ch3_temperature.png")
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[图] {os.path.relpath(out, REPO_ROOT)}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    imgs = sample_images()
    sim, scale = compute(imgs)
    fig_matrix(sim, [n.replace(".jpg", "").replace(".png", "") for n, _, _ in imgs])
    fig_temperature(sim, scale)


if __name__ == "__main__":
    main()
