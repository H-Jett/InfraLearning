#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 1 章练习：图像是怎么变成 LLM 能吃的 token 的。

四件事：
  A) 文本侧回顾：一句话 → tokenizer → token id → embedding 矩阵 [N, d]（你已熟悉的路径）；
  B) 图像侧手撕 patchify：一张图 → 切成 P×P 的 patch → 展平 → 线性投影 → [N, d]，
     并证明"unfold + Linear"与"stride=P 的 Conv2d"在数值上完全等价（patch embedding 的真身）；
  C) 真实 VLM（Qwen2.5-VL）：不同分辨率的图会产生多少视觉 token，公式与实测逐一对齐；
  D) infra 视角：这些视觉 token 进了 LLM，要占多少 KV cache、相当于多少字的文本。

只需要 config/processor（几百 KB），不下载模型权重，单卡/纯 CPU 都能跑。
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import sample_image  # noqa: E402

VLM_MODEL = os.environ.get("MM_VLM", "Qwen/Qwen2.5-VL-3B-Instruct")

SEP = "=" * 68


def human(n_bytes):
    n = float(n_bytes)
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if n < 1024 or unit == "GiB":
            return f"{n:.2f} {unit}"
        n /= 1024


# ---------------------------------------------------------------- A) 文本侧
def part_a(tokenizer, d_model):
    print(SEP)
    print("A) 文本侧回顾：句子 → token → embedding（你已经熟悉的路径）")
    print(SEP)
    text = "一只猫躺在沙发上。"
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    pieces = [tokenizer.decode([i]) for i in ids]
    print(f"原文        : {text}")
    print(f"token 数    : {len(ids)}")
    print(f"切分结果    : {pieces}")
    print(f"token id    : {ids}")
    print(f"embedding 后: [{len(ids)}, {d_model}] 的矩阵  ← 序列长度 × 隐藏维度")

    # 顺手量一下中文的"字/token"比值，D 部分要用它做等价换算（别拍脑袋）
    para = (
        "多模态模型的核心问题是把不同形态的信号统一到同一个表示空间里。"
        "对文本来说这件事已经被 tokenizer 解决了：一段话被切成若干 token，"
        "每个 token 查表得到一个向量。图像没有天然的词表，于是研究者选择了另一条路——"
        "把图像切成固定大小的小块，每一块经过一次线性变换成为一个向量。"
    )
    n_para = len(tokenizer(para, add_special_tokens=False)["input_ids"])
    ratio = len(para) / n_para
    print(f"\n（顺手一量）一段 {len(para)} 字的中文 = {n_para} 个 token"
          f"  → 约 {ratio:.2f} 汉字/token，D 部分做换算要用")

    print("\n关键：LLM 眼里的输入永远是 [序列长度, 隐藏维度] 的矩阵。")
    print("      图像要进来，也必须变成同样形状的东西——这就是下面 B 要做的事。")
    return len(ids), ratio


# ---------------------------------------------------------------- B) 手撕 patchify
def part_b(img):
    print("\n" + SEP)
    print("B) 图像侧：手撕 patchify（图像的 tokenizer）")
    print(SEP)

    P, D_MODEL, RES = 16, 768, 224   # patch 边长 / 投影维度 / 缩放到的分辨率
    arr = np.asarray(img.resize((RES, RES)), dtype=np.float32) / 255.0  # [H, W, 3]
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)             # [1, 3, 224, 224]
    print(f"输入张量     : {tuple(x.shape)}  (batch, 通道, 高, 宽)")

    # 1) 切 patch：unfold 把每个 P×P 窗口拉成一列
    patches = nn.functional.unfold(x, kernel_size=P, stride=P)  # [1, 3*P*P, N]
    patches = patches.transpose(1, 2)                           # [1, N, 3*P*P]
    n_patch = patches.shape[1]
    grid = RES // P
    print(f"patch 大小   : {P}×{P}")
    print(f"patch 网格   : {grid}×{grid} = {n_patch} 个 patch  ← 这就是图像的『序列长度』")
    print(f"每个 patch   : {P}×{P}×3 = {P * P * 3} 个数字，展平成一个向量")
    print(f"展平后       : {tuple(patches.shape)}")

    # 2) 线性投影到 d_model —— 等价于 kernel=stride=P 的 Conv2d
    torch.manual_seed(0)
    proj = nn.Linear(P * P * 3, D_MODEL, bias=True)
    tokens = proj(patches)  # [1, N, d_model]
    print(f"线性投影后   : {tuple(tokens.shape)}  ← [序列长度, 隐藏维度]，和文本 embedding 同形状！")

    conv = nn.Conv2d(3, D_MODEL, kernel_size=P, stride=P)
    with torch.no_grad():  # 用同一份权重，验证两种写法等价
        conv.weight.copy_(proj.weight.view(D_MODEL, 3, P, P))
        conv.bias.copy_(proj.bias)
        tokens_conv = conv(x).flatten(2).transpose(1, 2)
    max_diff = (tokens - tokens_conv).abs().max().item()
    print(f"\nConv2d(k={P}, s={P}) 的结果与 unfold+Linear 最大差异 = {max_diff:.2e}"
          f"  → {'完全等价 ✅' if max_diff < 1e-4 else '不等价 ❌'}")
    print("所以框架里那句 nn.Conv2d(3, d, kernel_size=P, stride=P) 就是 patch embedding，")
    print("它同时干了『切块』和『投影』两件事，没有任何玄学。")

    # 3) token 数随分辨率变化：面积关系
    print(f"\n分辨率 → token 数（patch={P}）：")
    print(f"{'分辨率':>12} | {'token 数':>9} | 相对 224²")
    print("-" * 40)
    base = (224 // P) ** 2
    for r in [224, 336, 448, 896]:
        n = (r // P) ** 2
        print(f"{f'{r}×{r}':>12} | {n:>9} | {n / base:>6.1f}×")
    print("规律：token 数 ∝ 像素数 ∝ 边长²。边长翻倍 → token 数 ×4 → 注意力算力 ×16。")
    return n_patch


# ---------------------------------------------------------------- C) 真实 VLM
def part_c(processor, img):
    print("\n" + SEP)
    print("C) 真实 VLM：Qwen2.5-VL 把一张图变成多少 token")
    print(SEP)

    ip = processor.image_processor
    p, merge = ip.patch_size, ip.merge_size
    unit = p * merge
    print(f"patch_size = {p}  merge_size = {merge}"
          f"  → 每 {unit}×{unit} 像素最终变成 1 个视觉 token")
    print(f"min_pixels = {ip.min_pixels}  max_pixels = {ip.max_pixels}"
          f"  （超出范围会先等比缩放）")

    print(f"\n（尺寸一律写成 宽×高）")
    print(f"{'输入尺寸':>12} | {'对齐后':>12} | {'grid(t,h,w)':>14} | {'公式':>6} | {'实测':>6} | 一致")
    print("-" * 76)
    results = []
    for size in [(224, 224), (336, 336), (640, 480), (960, 686), (1280, 960), (1920, 1080)]:
        im = img.resize(size)
        enc = ip(images=im, return_tensors="pt")
        t, h, w = enc["image_grid_thw"][0].tolist()
        formula = t * h * w // (merge * merge)

        # 实测：走完整对话模板，数一数 input_ids 里有多少个 <|image_pad|>
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": "描述这张图"}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[im], return_tensors="pt")
        pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        measured = int((inputs["input_ids"][0] == pad_id).sum())

        ok = "✅" if formula == measured else "❌"
        print(f"{f'{size[0]}×{size[1]}':>12} | {f'{w * p}×{h * p}':>12} |"
              f" {f'({t},{h},{w})':>14} | {formula:>6} | {measured:>6} | {ok}")
        results.append((size, measured))

    print(f"\n公式：视觉 token 数 = (H/{unit}) × (W/{unit})，其中 H、W 是缩放对齐后的高宽。")
    print("注意它和分辨率是**像素级线性**关系：像素翻倍，token 就翻倍——没有上限保护的话很危险，")
    print(f"所以 processor 用 max_pixels 兜底（{ip.max_pixels} 像素 ≈ "
          f"{ip.max_pixels // (unit * unit)} 个 token 的天花板）。")
    return results


# ---------------------------------------------------------------- D) infra 视角
def part_d(cfg, vision_results, n_text_tokens, chars_per_token):
    print("\n" + SEP)
    print("D) infra 视角：视觉 token 的代价")
    print(SEP)

    text_cfg = getattr(cfg, "text_config", cfg)
    L = text_cfg.num_hidden_layers
    n_kv = text_cfg.num_key_value_heads
    hidden = text_cfg.hidden_size
    head_dim = getattr(text_cfg, "head_dim", None) or hidden // text_cfg.num_attention_heads
    bpe = 2  # bf16
    per_token = 2 * L * n_kv * head_dim * bpe
    print(f"Qwen2.5-VL-3B 的 LLM 部分：层数={L}  KV头={n_kv}  head_dim={head_dim}  hidden={hidden}")
    print(f"每个 token 的 KV cache = 2×{L}×{n_kv}×{head_dim}×{bpe} = {human(per_token)}")

    print(f"\n（视觉 token 数直接用 C 部分的实测值，按 {chars_per_token:.2f} 汉字/token 换算文本等价量）")
    print(f"{'图像(宽×高)':>13} | {'视觉 token':>10} | {'KV cache':>10} | {'≈ 等量中文':>12}")
    print("-" * 56)
    for (w, h), n_tok in vision_results:
        print(f"{f'{w}×{h}':>13} | {n_tok:>10} | {human(n_tok * per_token):>10} |"
              f" {int(n_tok * chars_per_token):>10} 字")

    print(f"\n对比：本练习 A 里那句 {n_text_tokens} 个 token 的中文，KV cache 只有 "
          f"{human(n_text_tokens * per_token)}。")
    print("结论（贯穿全书的主线之一）：")
    print("  1. 一张图 = 几百到几千个 token：一张 1080p 图 ≈ 一篇几千字的文章；")
    print("  2. 这些 token 全部要过 prefill、全部要占 KV cache —— VLM 的 prefill 天然比纯文本重得多；")
    print("  3. 于是就有了后面章节的两条主线：")
    print("     · 架构上怎么『少出 token』（2×2 merge、Q-Former、token 压缩）；")
    print("     · 工程上怎么『扛住这些 token』（变长 packing、缓存复用、显存预算）。")


def main():
    from transformers import AutoConfig, AutoProcessor

    print(f"模型（只读 config/processor，不下权重）：{VLM_MODEL}\n")
    processor = AutoProcessor.from_pretrained(VLM_MODEL)
    cfg = AutoConfig.from_pretrained(VLM_MODEL)
    img = sample_image()
    print()

    text_cfg = getattr(cfg, "text_config", cfg)
    n_text, ratio = part_a(processor.tokenizer, text_cfg.hidden_size)
    part_b(img)
    vision_results = part_c(processor, img)
    part_d(cfg, vision_results, n_text, ratio)


if __name__ == "__main__":
    main()
