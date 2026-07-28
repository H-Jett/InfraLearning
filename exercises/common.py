#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
练习公用工具：准备一张示例图片。

优先从网络下载一张真实照片（缓存到 assets/sample/，已 gitignore）；
下载不通时回退到"程序生成的合成图"（彩色色块 + 圆 + 条纹），保证练习在任何环境都能跑。
"""

import os
import urllib.request
from PIL import Image, ImageDraw

# 相对仓库根的路径（本文件在 exercises/ 下）
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(REPO_ROOT, "assets", "sample")

# 一张 HuggingFace 文档示例图（960×686 的猫），仅用于演示
SAMPLE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/"
    "resolve/main/pipeline-cat-chonk.jpeg"
)
SAMPLE_NAME = "cat.jpg"


def _synthetic(path, size=(960, 686)):
    """回退方案：生成一张有结构的合成图（不同 patch 内容明显不同，便于观察）。"""
    img = Image.new("RGB", size, (240, 240, 240))
    d = ImageDraw.Draw(img)
    w, h = size
    d.rectangle([0, 0, w // 2, h // 2], fill=(0, 114, 178))          # 蓝块
    d.ellipse([w // 2 + 40, 40, w - 40, h // 2 - 40], fill=(230, 159, 0))  # 橙圆
    for i in range(0, w, 40):                                        # 下半部条纹
        d.rectangle([i, h // 2, i + 20, h], fill=(0, 158, 115))
    img.save(path)
    return img


def sample_image(verbose=True):
    """返回一张 PIL.Image（RGB）。优先用缓存 → 下载 → 合成图。"""
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    path = os.path.join(SAMPLE_DIR, SAMPLE_NAME)

    if not os.path.exists(path):
        try:
            if verbose:
                print("[图片] 本地无缓存，尝试下载示例图 ...")
            urllib.request.urlretrieve(SAMPLE_URL, path)
            if verbose:
                print(f"[图片] 下载完成：assets/sample/{SAMPLE_NAME}")
        except Exception as e:  # 网络不通就用合成图
            if verbose:
                print(f"[图片] 下载失败（{type(e).__name__}），改用程序生成的合成图")
            return _synthetic(path).convert("RGB")

    img = Image.open(path).convert("RGB")
    if verbose:
        print(f"[图片] 使用示例图 assets/sample/{SAMPLE_NAME}，尺寸 {img.size[0]}×{img.size[1]}")
    return img
