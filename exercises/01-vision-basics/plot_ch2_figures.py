#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成第 2 章的配图（输出到 docs/chapters/01-vision-basics/figures/）：

  ch2_attn_distance.png  —— 每个头的平均注意力距离随层数变化（12 层 × 12 头，实测）
  ch2_cls_attention.png  —— 最后一层 CLS 对各 patch 的注意力热力图，叠在原图上

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
from common import sample_image  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_DIR = os.path.join(REPO_ROOT, "docs", "chapters", "01-vision-basics", "figures")
BLUE, ORANGE, VERMILLION, GREEN = "#0072B2", "#E69F00", "#D55E00", "#009E73"

CLIP_MODEL = os.environ.get("MM_CLIP", "openai/clip-vit-base-patch32")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def forward(img):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    px = proc(images=img, return_tensors="pt").to(DEVICE)["pixel_values"]
    with torch.no_grad():
        out = model.vision_model(px, output_attentions=True)
    grid = model.config.vision_config.image_size // model.config.vision_config.patch_size
    return out.attentions, grid


def fig_attn_distance(attns, grid):
    """每个头的平均注意力距离 vs 层数——展示『层越深，局部选项越少』。"""
    ys, xs = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    coords = np.stack([ys.ravel(), xs.ravel()], axis=1).astype(np.float32)
    dist = torch.from_numpy(
        np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    ).to(attns[0].device)

    per_head = []
    for attn in attns:
        a = attn[0, :, 1:, 1:].float()
        a = a / a.sum(dim=-1, keepdim=True)
        per_head.append((a * dist).sum(dim=-1).mean(dim=-1).cpu().numpy())
    per_head = np.stack(per_head)                     # [layers, heads]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    n_layer, n_head = per_head.shape
    for L in range(n_layer):
        ax.scatter([L] * n_head, per_head[L], s=22, color=BLUE, alpha=0.55,
                   edgecolors="none", label="each attention head" if L == 0 else None)
    ax.plot(range(n_layer), per_head.mean(axis=1), color=VERMILLION, lw=1.8,
            marker="o", ms=4, label="layer mean")
    ax.plot(range(n_layer), per_head.min(axis=1), color=ORANGE, lw=1.3, ls="--",
            label="most local head")

    ax.set_xlabel("encoder layer")
    ax.set_ylabel("mean attention distance (patches)")
    ax.set_title("CLIP ViT-B/32: heads start polarized, end all-global", fontsize=12)
    ax.set_xticks(range(n_layer))
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "ch2_attn_distance.png")
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[图] {os.path.relpath(out, REPO_ROOT)}")


def fig_cls_attention(attns, grid, img):
    """最后一层 CLS→patch 的注意力热力图，叠在原图上。"""
    cls_attn = attns[-1][0, :, 0, 1:].float().mean(dim=0).cpu().numpy()  # 对 12 个头取平均
    heat = cls_attn.reshape(grid, grid)

    small = img.resize((224, 224))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))

    axes[0].imshow(small)
    axes[0].set_title("input (224x224)", fontsize=11)

    im = axes[1].imshow(heat, cmap="magma")
    axes[1].set_title(f"last-layer CLS attention\n({grid}x{grid} patches, mean of 12 heads)",
                      fontsize=10)
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    axes[2].imshow(small)
    axes[2].imshow(np.kron(heat, np.ones((224 // grid, 224 // grid))),
                   cmap="magma", alpha=0.55)
    axes[2].set_title("overlay: where the summary\ntoken looks", fontsize=10)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "ch2_cls_attention.png")
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[图] {os.path.relpath(out, REPO_ROOT)}")
    print(f"     （注意力最强的 patch 索引 {int(cls_attn.argmax())}，"
          f"最大/最小权重比 {cls_attn.max() / cls_attn.min():.1f}×）")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    img = sample_image()
    attns, grid = forward(img)
    fig_attn_distance(attns, grid)
    fig_cls_attention(attns, grid, img)


if __name__ == "__main__":
    main()
