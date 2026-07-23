#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
同步脚本：把源码(.py)和运行日志(.log)的真实内容，灌进各章 .md 附录里。

两类标记：
  <!-- CODE:exercises/PART/NN_name.py START -->      内联源码（唯一真源 = .py）
  <!-- OUTPUT:exercises/PART/logs/NN_name.log START --> 内联运行日志（唯一真源 = .log）
日志由 scripts/run_exercises.py 生成。

设计目标：
  - .py / .log 是唯一真源；.md 里存的是**真正的 fenced 块**（不是构建期引用），
    这样在 GitHub、任意 Markdown 阅读器、MkDocs 三边都能直接看到代码和输出。
  - 本脚本保证 .md 与源一致，杜绝复制粘贴导致的漂移。

用法：
  在 .md 里放一对标记（同一行注释）：

      <!-- CODE:exercises/01-inference/prefill_vs_decode.py START -->
      ```python
      （这里的内容会被脚本自动覆盖为 .py 的真实内容）
      ```
      <!-- CODE:exercises/01-inference/prefill_vs_decode.py END -->

  然后：
      python scripts/sync_code.py          # 把 .py 内容写进所有 .md 标记区
      python scripts/sync_code.py --check   # 只校验是否已同步；不一致则退出码 1（给 CI 用）

绝对路径：
  /volume/data/hjiang02/workspace/infra-learning/scripts/sync_code.py
"""

import argparse
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("sync_code")

REPO_ROOT = Path(__file__).resolve().parent.parent

# 匹配一对标记之间的整块内容。两类标记：
#   CODE:path   → 内联源码（.py 等），按后缀选语言
#   OUTPUT:path → 内联运行日志（.log），用 text 围栏
BLOCK_RE = re.compile(
    r"(<!-- (?P<kind>CODE|OUTPUT):(?P<path>[^\s]+) START -->)"
    r".*?"
    r"(<!-- (?P=kind):(?P=path) END -->)",
    re.DOTALL,
)

# .py 文件后缀 → fenced 代码块的语言标注
LANG = {".py": "python", ".sh": "bash", ".yaml": "yaml", ".yml": "yaml"}


def build_block(kind: str, rel_path: str) -> str:
    """根据标记类型和相对路径读取文件，生成 START..END 之间应有的完整文本。"""
    src = REPO_ROOT / rel_path
    if not src.exists():
        raise FileNotFoundError(f"{kind} 源文件不存在: {rel_path}（练习需先 run_exercises.py 生成日志）")
    lang = "text" if kind == "OUTPUT" else LANG.get(src.suffix, "")
    body = src.read_text(encoding="utf-8").rstrip("\n")
    return (
        f"<!-- {kind}:{rel_path} START -->\n"
        f"```{lang}\n{body}\n```\n"
        f"<!-- {kind}:{rel_path} END -->"
    )


def process(md_path: Path, check: bool) -> bool:
    """处理单个 .md。返回 True 表示该文件已（或已经）同步，False 表示 check 模式下发现不一致。"""
    text = md_path.read_text(encoding="utf-8")
    ok = True
    n = 0

    def repl(m: re.Match) -> str:
        nonlocal ok, n
        n += 1
        rel = m.group("path")
        want = build_block(m.group("kind"), rel)
        if m.group(0) != want:
            ok = False
            if not check:
                log.info("  同步: %s ← %s", md_path.relative_to(REPO_ROOT), rel)
        return want

    new_text = BLOCK_RE.sub(repl, text)
    if n == 0:
        return True
    if check:
        if not ok:
            log.warning("  未同步: %s", md_path.relative_to(REPO_ROOT))
        return ok
    if new_text != text:
        md_path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="同步 .py 源码到 .md 代码附录")
    ap.add_argument("--check", action="store_true", help="只校验，不写入；不一致则退出码 1")
    args = ap.parse_args()

    md_files = sorted((REPO_ROOT / "docs").rglob("*.md"))
    log.info("扫描 %d 个 .md（文档目录 %s/docs）", len(md_files), REPO_ROOT)

    all_ok = True
    for md in md_files:
        all_ok &= process(md, args.check)

    if args.check:
        if all_ok:
            log.info("校验通过：所有代码附录均已与源码同步。")
            return 0
        log.error("校验失败：有代码附录未同步，请运行 `python scripts/sync_code.py`。")
        return 1
    log.info("同步完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
