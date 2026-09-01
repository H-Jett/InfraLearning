#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建整个书架站点：每本书各自 mkdocs build 到 site/<slug>/，再生成索引页与 PWA。

设计要点：
  1. **books.yml 是单一真源**——构建哪些书、索引列哪些书、每本书什么配色，都从它读。
  2. **各书完全隔离**：各自的 mkdocs.yml、主题、搜索索引、校验脚本。
     本脚本只负责「按顺序调用各书自己的校验和构建」，不碰书的内部。
  3. **site_url 由 base_url 拼出**，通过环境变量 BOOK_SITE_URL 注入
     → **仓库改名时只改 books.yml 一行**。
  4. **配色下发**：每本书一套专属配色（books.yml 的 palette），由本脚本生成
     `books/<slug>/docs/assets/palette.css` 与 `books/<slug>/overrides/main.html`，
     保证「网页主题 / 索引卡片 / 手机地址栏」三处颜色一致；并**查重**，两本书不许撞色。
  5. **PWA**：书架级 manifest + service worker（scope 覆盖整个站点），
     手机可「添加到主屏幕」、独立窗口打开、离线可读。

生成物分两类：
  - **要提交**：books/<slug>/docs/assets/palette.css、books/<slug>/overrides/main.html
    （提交才能让 `cd books/tea && mkdocs serve` 单独调试时也是对的）
  - **不提交**：site/ 下的一切（manifest / sw.js / 索引页 / 各书站点）

用法：
    python scripts/build_site.py                # 全量构建
    python scripts/build_site.py --only tea     # 只构建某一本（可重复）
    python scripts/build_site.py --skip-checks  # 跳过各书校验，只构建
    python scripts/build_site.py --check        # 只校验（含生成物是否与 books.yml 同步），不构建
"""

import argparse
import hashlib
import html
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("build_site")

REPO_ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = REPO_ROOT / "site"
CONFIG = REPO_ROOT / "books.yml"
ICON_SRC = REPO_ROOT / "assets" / "icons"

GENERATED_BANNER = "由 scripts/build_site.py 依据 books.yml 生成，请勿手改"


# ----------------------------------------------------------------- 配色工具

def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c))):02X}" for c in rgb)


def mix(hex_color: str, target: str, f: float) -> str:
    """把颜色朝 target 混合 f（0~1）。用来生成 light/dark 变体。"""
    a, b = _rgb(hex_color), _rgb(target)
    return _hex(tuple(x + (y - x) * f for x, y in zip(a, b)))


def color_distance(c1: str, c2: str) -> float:
    """粗略的感知距离（加权欧氏）。够用来挡住「两本书撞色」。"""
    (r1, g1, b1), (r2, g2, b2) = _rgb(c1), _rgb(c2)
    rm = (r1 + r2) / 2
    dr, dg, db = r1 - r2, g1 - g2, b1 - b2
    return ((2 + rm / 256) * dr * dr + 4 * dg * dg + (2 + (255 - rm) / 256) * db * db) ** 0.5


MIN_DISTANCE = 120.0   # 经验阈值：低于此值两本书在小色块（索引卡片）上就不好分了


def check_palettes(books: list[dict]) -> None:
    """每本书必须有配色，且两两不撞。"""
    for b in books:
        if not b.get("palette", {}).get("primary"):
            raise ValueError(f"{b['slug']}: books.yml 里缺 palette.primary")
    for i, a in enumerate(books):
        for b in books[i + 1:]:
            d = color_distance(a["palette"]["primary"], b["palette"]["primary"])
            if d < MIN_DISTANCE:
                raise ValueError(
                    f"配色太接近：{a['slug']} ({a['palette']['primary']}) 与 "
                    f"{b['slug']} ({b['palette']['primary']}) 距离 {d:.0f} < {MIN_DISTANCE}。"
                    "每本书要有辨识度不同的配色，请换一个。")
    log.info("配色查重通过（%d 本书两两可区分）", len(books))


# ----------------------------------------------------------------- 每本书的生成物

PALETTE_CSS = """/* {banner} */
/* 《{title}》专属配色 —— 与索引页卡片、手机地址栏同一套值 */

:root,
[data-md-color-scheme="default"] {{
  --md-primary-fg-color:        {primary};
  --md-primary-fg-color--light: {primary_light};
  --md-primary-fg-color--dark:  {primary_dark};
  --md-primary-bg-color:        #ffffff;
  --md-primary-bg-color--light: #ffffffb3;
  --md-accent-fg-color:         {accent};
  --md-accent-fg-color--transparent: {accent}1a;
  --md-typeset-a-color:         {primary};
}}

[data-md-color-scheme="slate"] {{
  /* 深色底下主色要提亮，否则页头发闷、链接看不清 */
  --md-primary-fg-color:        {primary_dark};
  --md-primary-fg-color--light: {primary_light};
  --md-primary-fg-color--dark:  {primary_dark};
  --md-accent-fg-color:         {accent_light};
  --md-accent-fg-color--transparent: {accent}26;
  --md-typeset-a-color:         {accent_light};
}}
"""

OVERRIDES_MAIN = """{{% extends "base.html" %}}
<!-- {banner} -->

{{% block extrahead %}}
  <link rel="manifest" href="{base}manifest.webmanifest">
  <link rel="apple-touch-icon" href="{base}icons/apple-touch-icon-180.png">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <meta name="apple-mobile-web-app-title" content="{app_title}">
  <script>
    // 这本书的地址栏配色。Material 也会输出一条 theme-color，
    // 这里统一成本书的颜色，避免两条并存导致表现不一致。
    (function () {{
      function setThemeColor() {{
        document.querySelectorAll('meta[name="theme-color"]').forEach(function (m) {{
          m.parentNode.removeChild(m);
        }});
        var m = document.createElement('meta');
        m.name = 'theme-color';
        m.content = '{primary}';
        document.head.appendChild(m);
      }}
      setThemeColor();
      document.addEventListener('DOMContentLoaded', setThemeColor);
      if ('serviceWorker' in navigator) {{
        window.addEventListener('load', function () {{
          navigator.serviceWorker.register('{base}sw.js', {{ scope: '{base}' }});
        }});
      }}
    }})();
  </script>
{{% endblock %}}
"""


def book_generated_files(book: dict, base_path: str) -> dict[Path, str]:
    """这本书应该有哪些「由 books.yml 生成」的文件，以及它们应有的内容。"""
    p = book["palette"]
    primary, accent = p["primary"], p["accent"]
    book_dir = REPO_ROOT / book["dir"]
    # favicon 下发一份到每本书：这样 mkdocs.yml 里 theme.favicon 能指到它，
    # 由 Material 自己输出 <link rel="icon">，避免我们再插一条造成重复。
    favicon = (ICON_SRC / "favicon.svg").read_text(encoding="utf-8")
    return {
        book_dir / "docs" / "assets" / "favicon.svg": favicon,
        book_dir / "docs" / "assets" / "palette.css": PALETTE_CSS.format(
            banner=GENERATED_BANNER, title=book["title"],
            primary=primary,
            primary_light=mix(primary, "#FFFFFF", 0.28),
            primary_dark=mix(primary, "#000000", 0.22),
            accent=accent,
            accent_light=mix(accent, "#FFFFFF", 0.30),
        ),
        book_dir / "overrides" / "main.html": OVERRIDES_MAIN.format(
            banner=GENERATED_BANNER, base=base_path,
            app_title=book["title"], primary=primary,
        ),
    }


def sync_book_assets(books: list[dict], base_path: str, check: bool) -> bool:
    """写出（或校验）每本书的配色 CSS 与主题覆写。check 模式下不一致返回 False。"""
    ok = True
    for book in books:
        for path, want in book_generated_files(book, base_path).items():
            rel = path.relative_to(REPO_ROOT)
            have = path.read_text(encoding="utf-8") if path.exists() else None
            if have == want:
                continue
            if check:
                log.error("生成物与 books.yml 不同步: %s（跑一次 build_site.py 重新生成）", rel)
                ok = False
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(want, encoding="utf-8")
                log.info("  生成 %s", rel)
    return ok


# ----------------------------------------------------------------- 构建

def run(cmd: str, cwd: Path) -> None:
    log.info("  $ %s  (cwd=%s)", cmd, cwd.relative_to(REPO_ROOT))
    subprocess.run(cmd, cwd=cwd, shell=True, check=True)


def build_book(book: dict, base_url: str, skip_checks: bool, build: bool) -> None:
    book_dir = REPO_ROOT / book["dir"]
    if not (book_dir / "mkdocs.yml").exists():
        raise FileNotFoundError(f"{book['slug']}: 缺少 mkdocs.yml（{book_dir}）")

    log.info("[%s] %s", book["slug"], book["title"])
    if not skip_checks:
        for cmd in book.get("checks", []):
            run(cmd, book_dir)
    if not build:
        return

    out = SITE_DIR / book["slug"]
    env = dict(os.environ, BOOK_SITE_URL=f"{base_url}{book['slug']}/")
    log.info("  $ mkdocs build --strict  (BOOK_SITE_URL=%s)", env["BOOK_SITE_URL"])
    subprocess.run(["mkdocs", "build", "--strict", "-d", str(out)],
                   cwd=book_dir, env=env, check=True)


# ----------------------------------------------------------------- PWA

SERVICE_WORKER = """/* {banner} */
/* 书架 PWA 的 service worker。
   策略（这两条是自建 SW 最容易出错的地方，务必保持）：
     - **页面走 network-first**：内容更新后读者立刻能看到；断网才回落缓存。
     - **静态资源走 cache-first**：字体/图/JS 命中缓存，手机上翻页才够快。
   缓存名带版本号，激活时清掉旧版本，避免越积越多。 */

const VERSION = '{version}';
const BASE = '{base}';
const PAGES = 'pages-' + VERSION;
const STATIC = 'static-' + VERSION;
const PRECACHE = [BASE, BASE + 'manifest.webmanifest', BASE + 'icons/icon-192.png'];

self.addEventListener('install', (e) => {{
  e.waitUntil(caches.open(STATIC).then((c) => c.addAll(PRECACHE)).then(() => self.skipWaiting()));
}});

self.addEventListener('activate', (e) => {{
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((k) => !k.endsWith(VERSION)).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
}});

self.addEventListener('fetch', (e) => {{
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin || !url.pathname.startsWith(BASE)) return;

  if (req.mode === 'navigate') {{
    e.respondWith(
      fetch(req)
        .then((res) => {{
          const copy = res.clone();
          caches.open(PAGES).then((c) => c.put(req, copy));
          return res;
        }})
        .catch(() => caches.match(req).then((hit) => hit || caches.match(BASE)))
    );
    return;
  }}

  e.respondWith(
    caches.match(req).then((hit) => hit || fetch(req).then((res) => {{
      if (res.ok && res.type === 'basic') {{
        const copy = res.clone();
        caches.open(STATIC).then((c) => c.put(req, copy));
      }}
      return res;
    }}))
  );
}});
"""


def site_version() -> str:
    """用 site/ 下所有文件的(相对路径, 大小)算一个短哈希，内容变了版本才变。"""
    h = hashlib.sha256()
    for p in sorted(SITE_DIR.rglob("*")):
        if p.is_file() and p.name != "sw.js":
            h.update(str(p.relative_to(SITE_DIR)).encode())
            h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:12]


def emit_pwa(cfg: dict, base_path: str) -> None:
    if not ICON_SRC.exists():
        raise FileNotFoundError("缺少 assets/icons/，先本地跑 python scripts/make_icons.py")
    shutil.copytree(ICON_SRC, SITE_DIR / "icons", dirs_exist_ok=True)

    manifest = {
        "name": cfg.get("app_name", cfg.get("site_title", "")),
        "short_name": cfg.get("app_short_name", cfg.get("site_title", "")),
        "description": cfg.get("site_description", "").strip(),
        "start_url": base_path,
        "scope": base_path,
        "display": "standalone",
        "theme_color": cfg.get("app_theme_color", "#2C2C2A"),
        "background_color": cfg.get("app_background_color", "#FFFFFF"),
        "lang": "zh-CN",
        "icons": [
            {"src": base_path + "icons/icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": base_path + "icons/icon-512.png", "sizes": "512x512", "type": "image/png"},
            {"src": base_path + "icons/icon-maskable-512.png", "sizes": "512x512",
             "type": "image/png", "purpose": "maskable"},
        ],
        "shortcuts": [
            {"name": b["title"], "url": base_path + b["slug"] + "/"} for b in cfg["books"]
        ],
    }
    (SITE_DIR / "manifest.webmanifest").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (SITE_DIR / "sw.js").write_text(
        SERVICE_WORKER.format(banner=GENERATED_BANNER, version=site_version(), base=base_path),
        encoding="utf-8")
    log.info("PWA 就绪：manifest.webmanifest + sw.js + %d 个图标",
             len(list((SITE_DIR / "icons").glob("*.png"))))


# ----------------------------------------------------------------- 索引页

INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{site_title}</title>
<meta name="description" content="{site_description}">
<meta name="theme-color" content="{theme_color}">
<link rel="manifest" href="manifest.webmanifest">
<link rel="icon" href="icons/favicon.svg" type="image/svg+xml">
<link rel="alternate icon" href="icons/favicon-32.png" sizes="32x32">
<link rel="apple-touch-icon" href="icons/apple-touch-icon-180.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="{app_short_name}">
<style>
  :root {{
    --bg: #fbfbfa; --fg: #1b1b1a; --muted: #6a6a68;
    --card: #ffffff; --line: #e6e6e3; --shadow: 0 1px 2px rgba(0,0,0,.05);
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #16171a; --fg: #e8e8e6; --muted: #9a9a97;
      --card: #1e1f23; --line: #2c2d32; --shadow: none;
    }}
  }}
  * {{ box-sizing: border-box; -webkit-tap-highlight-color: transparent; }}
  body {{
    margin: 0; background: var(--bg); color: var(--fg);
    font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans CJK SC",
          "Source Han Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
    -webkit-font-smoothing: antialiased;
    padding: env(safe-area-inset-top) env(safe-area-inset-right)
             env(safe-area-inset-bottom) env(safe-area-inset-left);
  }}
  .wrap {{ max-width: 860px; margin: 0 auto; padding: 56px 20px 80px; }}
  header {{ margin-bottom: 36px; }}
  h1 {{ font-size: 28px; margin: 0 0 8px; letter-spacing: -.01em; }}
  .tagline {{ color: var(--muted); font-size: 15px; margin: 0 0 14px; }}
  .desc {{ color: var(--muted); font-size: 14.5px; max-width: 62ch; margin: 0; }}
  .grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); }}
  a.card {{
    display: block; text-decoration: none; color: inherit; background: var(--card);
    border: 1px solid var(--line); border-radius: 12px; padding: 20px 20px 18px;
    box-shadow: var(--shadow); transition: transform .12s ease, border-color .12s ease;
    position: relative; overflow: hidden;
  }}
  a.card:active {{ transform: scale(.985); }}
  @media (hover: hover) {{ a.card:hover {{ transform: translateY(-2px); border-color: var(--accent); }} }}
  a.card::before {{
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent);
  }}
  .card h2 {{ font-size: 19px; margin: 0 0 4px; }}
  .card .sub {{ color: var(--muted); font-size: 13.5px; margin: 0 0 10px; }}
  .card p {{ font-size: 14px; margin: 0 0 12px; }}
  .meta {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-size: 12.5px; }}
  .badge {{
    border: 1px solid var(--accent); color: var(--accent);
    border-radius: 999px; padding: 1px 9px; font-weight: 600;
  }}
  .progress {{ color: var(--muted); }}
  footer {{ margin-top: 48px; color: var(--muted); font-size: 13px; }}
  footer a {{ color: inherit; }}
  @media (max-width: 480px) {{
    .wrap {{ padding: 40px 16px 64px; }}
    h1 {{ font-size: 24px; }}
    .grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>{site_title}</h1>
    <p class="tagline">{site_tagline}</p>
    <p class="desc">{site_description}</p>
  </header>
  <main class="grid">
{cards}
  </main>
  <footer>
    源码与勘误：<a href="{repo_url}">{repo_short}</a>
  </footer>
</div>
<script>
  if ('serviceWorker' in navigator) {{
    window.addEventListener('load', function () {{
      navigator.serviceWorker.register('sw.js', {{ scope: '{base_path}' }});
    }});
  }}
</script>
</body>
</html>
"""

CARD_TEMPLATE = """    <a class="card" href="{slug}/" style="--accent: {accent}">
      <h2>{title}</h2>
      <p class="sub">{subtitle}</p>
      <p>{description}</p>
      <div class="meta">
        <span class="badge">{status}</span>
        <span class="progress">{progress}</span>
      </div>
    </a>"""


def render_index(cfg: dict, base_path: str) -> None:
    e = lambda s: html.escape(str(s), quote=True)
    cards = "\n".join(
        CARD_TEMPLATE.format(
            slug=e(b["slug"]), accent=e(b["palette"]["primary"]),
            title=e(b["title"]), subtitle=e(b.get("subtitle", "")),
            description=e(b.get("description", "")),
            status=e(b.get("status", "")), progress=e(b.get("progress", "")),
        )
        for b in cfg["books"]
    )
    repo_url = cfg.get("repo_url", "")
    page = INDEX_TEMPLATE.format(
        site_title=e(cfg.get("site_title", "书架")),
        site_tagline=e(cfg.get("site_tagline", "")),
        site_description=e(cfg.get("site_description", "").strip()),
        theme_color=e(cfg.get("app_theme_color", "#2C2C2A")),
        app_short_name=e(cfg.get("app_short_name", cfg.get("site_title", ""))),
        repo_url=e(repo_url), repo_short=e(repo_url.replace("https://github.com/", "")),
        base_path=e(base_path), cards=cards,
    )
    (SITE_DIR / "index.html").write_text(page, encoding="utf-8")
    (SITE_DIR / ".nojekyll").write_text("", encoding="utf-8")
    log.info("索引页已生成: site/index.html（%d 本书）", len(cfg["books"]))


# ----------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description="构建书架站点")
    ap.add_argument("--only", action="append", metavar="SLUG", help="只处理某本书（可重复）")
    ap.add_argument("--skip-checks", action="store_true", help="跳过各书自己的校验")
    ap.add_argument("--check", action="store_true", help="只校验（含生成物同步），不构建")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if not cfg["base_url"].endswith("/"):
        cfg["base_url"] += "/"
    base_url = cfg["base_url"]
    base_path = urlparse(base_url).path or "/"

    check_palettes(cfg["books"])

    books = cfg["books"]
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {b["slug"] for b in books}
        if unknown:
            log.error("books.yml 里没有这些书: %s", ", ".join(sorted(unknown)))
            return 1
        books = [b for b in books if b["slug"] in wanted]

    # 配色 CSS 与主题覆写：check 模式只比对，不写盘
    if not sync_book_assets(cfg["books"], base_path, check=args.check):
        return 1

    build = not args.check
    if build and not args.only and SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)     # 全量构建先清干净，避免删掉的书残留在站点里
    SITE_DIR.mkdir(parents=True, exist_ok=True)

    log.info("base_url = %s", base_url)
    for b in books:
        build_book(b, base_url, args.skip_checks, build)

    if build:
        render_index(cfg, base_path)   # 索引始终按 books.yml 全量渲染，即使只构建了一本
        emit_pwa(cfg, base_path)
        log.info("完成：site/ 下 %d 本书 + 索引页 + PWA", len(cfg["books"]))
    else:
        log.info("校验通过：%d 本书", len(books))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        if isinstance(e, subprocess.CalledProcessError):
            log.error("子命令失败（退出码 %s）: %s", e.returncode, e.cmd)
        else:
            log.error("%s", e)
        sys.exit(1)
