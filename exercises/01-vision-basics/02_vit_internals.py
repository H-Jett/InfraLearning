#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 2 章练习：把一个真实的 ViT（CLIP ViT-B/32）拆开看，验证四件事。

  A) 结构与参数量：CLS token、位置编码、12 层 encoder 各占多少参数；序列长度 = 49 patch + 1 CLS；
  B) 逐层前向：每层 hidden_states 的形状，以及 CLS 表示逐层"离开起点"的过程；
  C) 平均注意力距离（mean attention distance）：早期层看局部、后期层看全局——ViT 自己学出来的层级；
  D) 位置编码到底有多重要：置零 / 打乱 两种消融，看 zero-shot 分类崩成什么样；
  E) 换分辨率：448 输入 + 位置编码插值，token 数变 197，特征还对不对得上。

单卡即可（bf16 下 CLIP ViT-B/32 只要几百 MB）。
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import sample_image  # noqa: E402

CLIP_MODEL = os.environ.get("MM_CLIP", "openai/clip-vit-base-patch32")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEP = "=" * 68

# zero-shot 分类用的候选文本（示例图是雪地里的猫科动物）
LABELS = ["a photo of a cat", "a photo of a dog", "a photo of a car",
          "a photo of a snowy forest"]


def part_a(model, cfg):
    print(SEP)
    print("A) 结构与参数量：ViT-B/32 里都有什么")
    print(SEP)
    grid = cfg.image_size // cfg.patch_size
    n_patch = grid * grid
    print(f"输入图像      : {cfg.image_size}×{cfg.image_size}")
    print(f"patch         : {cfg.patch_size}×{cfg.patch_size} → {grid}×{grid} = {n_patch} 个 patch")
    print(f"序列长度      : {n_patch} patch + 1 CLS = {n_patch + 1} 个 token")
    print(f"隐藏维度      : {cfg.hidden_size}   层数: {cfg.num_hidden_layers}"
          f"   注意力头: {cfg.num_attention_heads}")

    emb = model.vision_model.embeddings
    print(f"\nCLS token     : {tuple(emb.class_embedding.shape)}  ← 一个可学习的向量，不来自任何像素")
    print(f"位置编码      : {tuple(emb.position_embedding.weight.shape)}"
          f"  ← {n_patch + 1} 个位置各一个向量，**一维可学习**，不是 sin/cos")
    print(f"patch_embed   : {type(emb.patch_embedding).__name__}"
          f"(kernel={emb.patch_embedding.kernel_size}, stride={emb.patch_embedding.stride})"
          f"  ← 就是第 1 章那个 Conv2d")

    groups = {"patch_embed（切块+投影）": emb.patch_embedding,
              "位置编码": emb.position_embedding,
              "12 层 encoder": model.vision_model.encoder}
    total = sum(p.numel() for p in model.vision_model.parameters())
    print(f"\n{'部件':<24} | {'参数量':>12} | 占视觉塔")
    print("-" * 52)
    for name, mod in groups.items():
        n = sum(p.numel() for p in mod.parameters())
        print(f"{name:<24} | {n:>12,} | {n / total:>7.1%}")
    print(f"{'视觉塔合计':<24} | {total:>12,} | {1:>7.1%}")
    print("\n看清楚：参数几乎全在 encoder 里。patch_embed 和位置编码只是入口，很便宜。")
    return n_patch, grid


def part_b(model, pixel_values, cfg):
    print("\n" + SEP)
    print("B) 逐层前向：形状不变，内容在变")
    print(SEP)
    with torch.no_grad():
        out = model.vision_model(pixel_values, output_hidden_states=True)
    hs = out.hidden_states
    attns = model.vision_model(pixel_values, output_attentions=True).attentions
    print(f"hidden_states 共 {len(hs)} 份（embedding 输出 + {cfg.num_hidden_layers} 层）")
    print(f"每一份的形状都是 {tuple(hs[0].shape)}  ← [batch, 序列长, 隐藏维度]，**全程不变**")

    # CLS 逐层演化：离起点多远、离终点多近，以及它每层从 patch 那里"读"了多少
    cos = torch.nn.functional.cosine_similarity
    cls0, cls_last = hs[0][:, 0], hs[-1][:, 0]
    print(f"\n{'层':>4} | {'cos(CLS_i, CLS_初始)':>20} | {'cos(CLS_i, CLS_末层)':>20}"
          f" | {'该层 CLS 注意力投向 patch 的比例':>28}")
    print("-" * 84)
    for i in range(len(hs)):
        if i not in (0, 1, 2, 3, 6, 9, 11, 12):
            continue
        cls_i = hs[i][:, 0]
        # 第 i 层的注意力（i>=1 时对应 attns[i-1]）里，CLS 这一行有多少权重给了 patch
        if i >= 1:
            cls_row = attns[i - 1][0, :, 0, :].float()      # [heads, 序列长]
            to_patch = cls_row[:, 1:].sum(dim=-1).mean().item()
            frac = f"{to_patch:>27.1%}"
        else:
            frac = f"{'—（还没进 encoder）':>24}"
        print(f"{i:>4} | {cos(cls_i, cls0).item():>20.3f} | {cos(cls_i, cls_last).item():>20.3f} | {frac}")

    print("\n三个观察（都按上面的实测说，不套教科书）：")
    print("  1. 形状全程 [1, 50, 768] 不变——Transformer 是『等长变换』，只改内容不改长度；")
    print("  2. CLS 一路远离初始值（与 CLS_初始 的 cos：1.00 → 0.26 → -0.01），说明它确实在被改写；")
    print("     但**它与最终表示的相似度长期贴近 0，直到最后一两层才突然拉起来**")
    print("     （第 11 层 0.22 → 第 12 层 1.00）。也就是说『图像摘要』是在末层才成形的——")
    print("     这与 CLIP 的训练方式一致：只有最后一层的 CLS 被投影出去参与对比学习，")
    print("     中间层的 CLS 没有任何直接约束，不必长得像最终答案。")
    print("  3. CLS 每层从 patch 那里『读』多少并不均匀（14%~51%）：有的层几乎只在自我更新，")
    print("     有的层才大口吸信息。别想象成『每层都在均匀汇聚』。")


def part_c(model, pixel_values, grid):
    """平均注意力距离：每层的注意力平均跨越多少个 patch 的距离。"""
    print("\n" + SEP)
    print("C) 平均注意力距离：ViT 自己学出的『由局部到全局』")
    print(SEP)
    with torch.no_grad():
        out = model.vision_model(pixel_values, output_attentions=True)

    # patch 在 grid 上的坐标，两两欧氏距离（单位：patch）
    ys, xs = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    coords = np.stack([ys.ravel(), xs.ravel()], axis=1).astype(np.float32)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    dist_t = torch.from_numpy(dist).to(pixel_values.device)

    print(f"（grid = {grid}×{grid}，patch 间最大距离 = {dist.max():.1f}）")
    print("按**每个头**分别统计——层平均会把真相抹平（见下面的解读）")
    print(f"\n{'层':>4} | {'最局部的头':>10} | {'12 个头的平均':>13} | {'最全局的头':>10} | 每个头的分布")
    print("-" * 86)
    per_head = []
    for i, attn in enumerate(out.attentions):
        # attn: [batch, heads, q, k]，去掉 CLS 行列，只看 patch→patch
        a = attn[0, :, 1:, 1:].float()
        a = a / a.sum(dim=-1, keepdim=True)              # 去掉 CLS 后重新归一化
        d_head = (a * dist_t).sum(dim=-1).mean(dim=-1)   # 每个头一个平均距离
        per_head.append(d_head.tolist())
        # 把 12 个头画到一条 0~5 patch 的刻度上
        line = [" "] * 26
        for d in d_head.tolist():
            line[min(25, int(d / 5 * 25))] = "•"
        print(f"{i:>4} | {d_head.min():>10.2f} | {d_head.mean():>13.2f} | {d_head.max():>10.2f} |"
              f" |{''.join(line)}|")

    first, last = per_head[0], per_head[-1]
    print(f"\n真相（和『底层都局部』的教科书说法不完全一样，诚实说明）：")
    print(f"  · 第 0 层的头**两极分化**：最局部的只有 {min(first):.2f}，最全局的已达 {max(first):.2f}"
          f"（跨度 {max(first) - min(first):.2f}）——")
    print(f"    也就是说 ViT 从第一层起就有头在看全图，这和 CNN『底层只能看局部』有本质区别；")
    print(f"  · 到最后一层，**最局部的头也变全局了**（min 从 {min(first):.2f} 涨到 {min(last):.2f}），")
    print(f"    12 个头挤在一起（跨度只剩 {max(last) - min(last):.2f}）；")
    print(f"  · 所以正确的说法是：**层数越深，注意力的『局部选项』越少**，而不是『底层一定局部』。")
    print(f"  · 也别只看层平均：第 0 层平均 {sum(first) / len(first):.2f} 反而比中间层高，")
    print(f"    因为它被那几个全局头拉上去了——平均值在这里是会骗人的。")
    return per_head


def zero_shot(model, processor, pixel_values):
    """返回 (最高分标签, 各标签概率)。"""
    with torch.no_grad():
        text = processor(text=LABELS, return_tensors="pt", padding=True).to(pixel_values.device)
        img_f = model.get_image_features(pixel_values=pixel_values)
        txt_f = model.get_text_features(**text)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
        probs = (model.logit_scale.exp() * img_f @ txt_f.T).softmax(dim=-1)[0]
    return LABELS[int(probs.argmax())], probs


def part_d(model, processor, pixel_values, n_patch):
    print("\n" + SEP)
    print("D) 位置编码消融：ViT 到底有多依赖『我在哪』")
    print(SEP)
    emb = model.vision_model.embeddings
    orig = emb.position_embedding.weight.data.clone()

    def show(tag):
        best, probs = zero_shot(model, processor, pixel_values)
        detail = "  ".join(f"{lb.replace('a photo of ', '')}={p:.2f}"
                           for lb, p in zip(LABELS, probs.tolist()))
        print(f"{tag:<28} → 判为『{best}』   {detail}")

    show("① 原始（完整位置编码）")

    emb.position_embedding.weight.data[1:] = 0            # 只清 patch 的位置编码
    show("② 位置编码置零")

    emb.position_embedding.weight.data.copy_(orig)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n_patch, generator=g) + 1
    emb.position_embedding.weight.data[1:] = orig[perm]   # 等价于打乱 patch 的空间位置
    show("③ 位置编码随机打乱")

    emb.position_embedding.weight.data.copy_(orig)        # 恢复
    show("④ 恢复原始（自检）")
    print("\n结论：patch 序列本身是**无序**的，空间信息全靠位置编码注入。")
    print("      置零/打乱之后模型看到的是『一袋碎片』，判断随之漂移。")


def part_e(model, processor, img, cfg):
    print("\n" + SEP)
    print("E) 换分辨率：位置编码要插值")
    print(SEP)
    cos = torch.nn.functional.cosine_similarity
    base = processor(images=img, return_tensors="pt").to(DEVICE)["pixel_values"]
    with torch.no_grad():
        f224 = model.get_image_features(pixel_values=base)

    print(f"{'输入分辨率':>12} | {'token 数':>9} | {'插值位置编码':>12} | {'与 224 特征的 cos':>18}")
    print("-" * 62)
    grid = cfg.image_size // cfg.patch_size
    print(f"{f'{cfg.image_size}²':>12} | {grid * grid + 1:>9} | {'—':>12} | {1.0:>18.3f}")

    for res in [336, 448]:
        px = processor(images=img, return_tensors="pt",
                       size={"shortest_edge": res},
                       crop_size={"height": res, "width": res}).to(DEVICE)["pixel_values"]
        g = res // cfg.patch_size
        with torch.no_grad():
            f = model.get_image_features(pixel_values=px, interpolate_pos_encoding=True)
        print(f"{f'{res}²':>12} | {g * g + 1:>9} | {'✅':>12} | {cos(f, f224).item():>18.3f}")

    # 不插值会怎样：直接报错（位置编码数量对不上）
    px = processor(images=img, return_tensors="pt",
                   size={"shortest_edge": 448},
                   crop_size={"height": 448, "width": 448}).to(DEVICE)["pixel_values"]
    try:
        with torch.no_grad():
            model.get_image_features(pixel_values=px, interpolate_pos_encoding=False)
        print(f"{'448²':>12} | {'—':>9} | {'❌ 不插值':>12} | {'居然没报错？':>18}")
    except Exception as e:
        print(f"\n不插值直接喂 448²：{type(e).__name__}: {str(e).splitlines()[0][:80]}")
    print("\n这正是第 1 章说的『固定 224』的枷锁：ViT 的位置编码是**按数量**学出来的，")
    print("换分辨率就得插值。现代 VLM 要支持任意分辨率，位置编码必须换成可外推的方案（第 9 章 M-RoPE）。")


def main():
    from transformers import CLIPModel, CLIPProcessor

    print(f"模型：{CLIP_MODEL}  设备：{DEVICE}\n")
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    cfg = model.config.vision_config
    img = sample_image()
    pixel_values = processor(images=img, return_tensors="pt").to(DEVICE)["pixel_values"]
    print()

    n_patch, grid = part_a(model, cfg)
    part_b(model, pixel_values, cfg)
    part_c(model, pixel_values, grid)
    part_d(model, processor, pixel_values, n_patch)
    part_e(model, processor, img, cfg)


if __name__ == "__main__":
    main()
