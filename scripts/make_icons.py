#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成站点图标：🧐（U+1F9D0）叠在深色圆角底上。

**在本地跑，产物提交进仓库**——CI 里不跑。
理由：栅格化依赖 cairosvg + 系统 libcairo2，不想让发布流程多一个系统级依赖；
图标很少变，committed 是更稳的选择。

素材：Noto Emoji 的矢量 🧐（Apache-2.0），已归档在 assets/icons/source/，
出处见同目录 PROVENANCE.md。**不依赖本机字体**——本机一个字体都没有，
靠渲染 emoji 文字必然出方框，所以走矢量素材。

用法：
    python scripts/make_icons.py            # 生成 assets/icons/*
    python scripts/make_icons.py --check    # 只校验产物齐备（CI 用）
"""

import argparse
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("make_icons")

REPO_ROOT = Path(__file__).resolve().parent.parent
ICON_DIR = REPO_ROOT / "assets" / "icons"
EMOJI_SRC = ICON_DIR / "source" / "emoji-1f9d0-noto.svg"

BG = "#2C2C2A"          # 深墨底：让 🧐 的黄色跳出来，且与 books.yml 的 app_theme_color 一致

# (文件名, 边长, 内容占比)
# maskable 会被系统裁成圆形，内容必须缩在中间安全区内 → 占比更小。
PNGS = [
    ("icon-192.png", 192, 0.74),
    ("icon-512.png", 512, 0.74),
    ("icon-maskable-512.png", 512, 0.56),
    ("apple-touch-icon-180.png", 180, 0.74),
    ("favicon-32.png", 32, 0.86),
]


def emoji_inner() -> tuple[str, str]:
    """取出 emoji SVG 的 viewBox 与内部内容（去掉 XML 声明与注释）。"""
    raw = EMOJI_SRC.read_text(encoding="utf-8")
    vb = re.search(r'viewBox="([^"]+)"', raw)
    if not vb:
        raise ValueError(f"{EMOJI_SRC} 里找不到 viewBox")
    body = raw[raw.index(">", raw.index("<svg")) + 1: raw.rindex("</svg>")]
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    return vb.group(1), body


def compose(size: int, scale: float, rounded: bool = True) -> str:
    """底色圆角方块 + 居中的 🧐。用嵌套 <svg> 缩放，浏览器与 cairosvg 都支持。"""
    vb, body = emoji_inner()
    inner = size * scale
    off = (size - inner) / 2
    r = size * 0.22 if rounded else 0
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">'
        f'<title>边学边记</title>'
        f'<rect x="0" y="0" width="{size}" height="{size}" rx="{r:.1f}" fill="{BG}"/>'
        f'<svg x="{off:.2f}" y="{off:.2f}" width="{inner:.2f}" height="{inner:.2f}" '
        f'viewBox="{vb}">{body}</svg>'
        f'</svg>'
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="生成站点图标")
    ap.add_argument("--check", action="store_true", help="只校验产物齐备")
    args = ap.parse_args()

    wanted = [n for n, _, _ in PNGS] + ["favicon.svg"]
    if args.check:
        missing = [n for n in wanted if not (ICON_DIR / n).exists()]
        if missing:
            log.error("缺少图标: %s（本地跑 python scripts/make_icons.py 生成并提交）",
                      ", ".join(missing))
            return 1
        log.info("图标齐备（%d 个）。", len(wanted))
        return 0

    if not EMOJI_SRC.exists():
        log.error("缺少素材 %s（见 assets/icons/source/PROVENANCE.md）",
                  EMOJI_SRC.relative_to(REPO_ROOT))
        return 1
    try:
        import cairosvg
    except ImportError:
        log.error("需要 cairosvg（且系统要有 libcairo2）："
                  "pip install cairosvg / apt-get install -y libcairo2")
        return 1

    ICON_DIR.mkdir(parents=True, exist_ok=True)
    for name, size, scale in PNGS:
        out = ICON_DIR / name
        cairosvg.svg2png(bytestring=compose(size, scale).encode("utf-8"),
                         write_to=str(out), output_width=size, output_height=size)
        log.info("生成 %s (%dx%d)", out.relative_to(REPO_ROOT), size, size)

    # 矢量 favicon：自包含（内嵌路径，不依赖读者本机的 emoji 字体），任意尺寸都清晰
    (ICON_DIR / "favicon.svg").write_text(compose(64, 0.86), encoding="utf-8")
    log.info("生成 %s", (ICON_DIR / "favicon.svg").relative_to(REPO_ROOT))
    log.info("完成。图标是**生成物但要提交**，CI 不重新生成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
