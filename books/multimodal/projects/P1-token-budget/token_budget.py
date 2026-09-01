#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实战项目 1：图像 token 预算计算器 —— 骨架代码（你要填的地方标了 TODO）

任务说明见 docs/projects/P1-token-budget.md。
验收：`python check.py`（对照 transformers 的真实 processor 打分）。

四个要实现的函数：
    1. smart_resize()           复现 Qwen-VL 的预处理尺寸对齐逻辑
    2. vision_tokens()          原图宽高 → 送进 LLM 的视觉 token 数
    3. kv_cache_bytes()         token 数 → KV cache 字节数
    4. max_pixels_for_budget()  反向：显存预算 → 最大允许 max_pixels

约定（别改，check.py 按这个签名调用）：
    - 所有尺寸参数都是 (height, width) 顺序，返回值也是 (height, width)；
    - factor = patch_size × merge_size（Qwen2.5-VL 是 14 × 2 = 28）；
    - min_pixels / max_pixels 约束的是**像素总数**，不是边长。
"""

import math


# --------------------------------------------------------------------------- 1
def smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=12845056):
    """把 (height, width) 调整成能被 factor 整除、且像素总数落在 [min_pixels, max_pixels] 的尺寸。

    要求：
      - 尽量保持原始长宽比；
      - 返回的 h、w 都是 factor 的整数倍，且都 >= factor；
      - factor 对齐后如果像素数超过 max_pixels，要等比缩小到不超过（注意取整方向）；
      - 如果像素数小于 min_pixels，要等比放大到不小于（注意取整方向）；
      - 极端长宽比（如 4000×50）也不能崩。

    Args:
        height, width: 原图高、宽（像素）
        factor: 对齐单位（patch_size × merge_size）
        min_pixels, max_pixels: 像素总数的下限与上限

    Returns:
        (new_height, new_width)：都是 factor 的整数倍

    提示：先各自四舍五入到 factor 倍数，再处理超限/不足两种情况。
         缩小时向下取整、放大时向上取整——想清楚为什么。
    """
    # TODO(1): 实现它
    raise NotImplementedError("TODO(1): smart_resize")


# --------------------------------------------------------------------------- 2
def vision_tokens(height, width, factor=28, merge_size=2,
                  min_pixels=56 * 56, max_pixels=12845056):
    """给定原图尺寸，返回**真正送进 LLM** 的视觉 token 数。

    注意：不是 patch 数。patch 边长是 factor / merge_size，而相邻 merge_size×merge_size
    个 patch 会被合并成 1 个 token。

    Returns:
        int：视觉 token 数
    """
    # TODO(2): 实现它（先调用 smart_resize，再算 token 数）
    raise NotImplementedError("TODO(2): vision_tokens")


# --------------------------------------------------------------------------- 3
def kv_cache_bytes(n_tokens, num_layers, num_kv_heads, head_dim, bytes_per_elem=2):
    """这些 token 的 KV cache 占多少字节。

    Args:
        n_tokens: token 数（视觉 + 文本都算）
        num_layers: LLM 层数
        num_kv_heads: KV 头数（GQA 下小于 attention 头数，别搞错）
        head_dim: 每个头的维度
        bytes_per_elem: 每个元素字节数（bf16/fp16 = 2，fp8 = 1）

    Returns:
        int：字节数
    """
    # TODO(3): 实现它（第 1 章的公式，K 和 V 各存一份）
    raise NotImplementedError("TODO(3): kv_cache_bytes")


# --------------------------------------------------------------------------- 4
def max_pixels_for_budget(budget_bytes, concurrency, num_layers, num_kv_heads, head_dim,
                          bytes_per_elem=2, factor=28, merge_size=2, text_tokens=0):
    """反向求解：给定 KV cache 显存预算，每张图最多能有多少像素？

    场景：你要服务 `concurrency` 路并发，每路 1 张图 + `text_tokens` 个文本 token，
    整个 KV cache 不能超过 `budget_bytes`。问 max_pixels 该设成多少。

    要求：
      - 返回的 max_pixels 代回 vision_tokens() 计算，总显存必须 <= budget_bytes；
      - 但也不能过于保守：必须 >= 预算的 90%（把预算用起来）；
      - 返回值应是 factor² 的整数倍（否则多出的像素凑不满一个 token，没意义）。

    Returns:
        int：推荐的 max_pixels
    """
    # TODO(4): 实现它（解析解或二分都行，check.py 只看结果）
    raise NotImplementedError("TODO(4): max_pixels_for_budget")


# --------------------------------------------------------------------------- 自测
def _demo():
    """填完上面四个函数后，跑 `python token_budget.py` 看一眼直观结果。"""
    print("原图 → 对齐后 → 视觉 token 数")
    for h, w in [(224, 224), (480, 640), (686, 960), (1080, 1920), (6000, 8000)]:
        rh, rw = smart_resize(h, w)
        n = vision_tokens(h, w)
        print(f"  {w:>5}×{h:<5} → {rw:>5}×{rh:<5} → {n:>6} tokens")

    # Qwen2.5-VL-3B 的 LLM 配置
    cfg = dict(num_layers=36, num_kv_heads=2, head_dim=128)
    n = vision_tokens(1080, 1920)
    print(f"\n1080p 一张图的 KV cache = {kv_cache_bytes(n, **cfg) / 2**20:.2f} MiB")

    mp = max_pixels_for_budget(8 * 2**30, concurrency=32, text_tokens=200, **cfg)
    print(f"8 GiB 预算 / 32 路并发 → 推荐 max_pixels = {mp}"
          f"（≈ {mp // (28 * 28)} tokens 上限）")


if __name__ == "__main__":
    _demo()
