#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行每个练习脚本，把它的 stdout 存成输出文件，作为章节里"运行输出"的唯一真源。

  - 扫描 exercises/*/NN_name.py（两位数字前缀的练习，不含 plot_*）；
  - 逐个用 `python` 跑（CUDA_VISIBLE_DEVICES=0），捕获 stdout；
  - 存到 exercises/PART/outputs/NN_name.txt；
  - 失败时把 stderr 末尾也追加进日志，方便排查。

日志更新是**主动**行为：改了练习代码后手动跑一次本脚本刷新日志，再 sync 到章节。

用法：
  python scripts/run_exercises.py                # 跑全部
  python scripts/run_exercises.py 03 07          # 只跑编号 03、07

绝对路径：
  /volume/data/hjiang02/workspace/infra-learning/scripts/run_exercises.py
"""

import os
import re
import sys
import glob
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMEOUT = 600  # 每个练习最多跑 10 分钟


def main():
    only = set(sys.argv[1:])  # 可选：只跑指定编号，如 "03"
    targets = sorted(glob.glob(os.path.join(REPO, "exercises", "*", "[0-9][0-9]_*.py")))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONWARNINGS="ignore")
    print(f"共 {len(targets)} 个练习待运行\n")

    for py in targets:
        num = re.match(r"(\d\d)_", os.path.basename(py)).group(1)
        if only and num not in only:
            continue
        logdir = os.path.join(os.path.dirname(py), "outputs")
        os.makedirs(logdir, exist_ok=True)
        log = os.path.join(logdir, os.path.basename(py).replace(".py", ".txt"))
        rel = os.path.relpath(py, REPO)
        print(f"[跑] {rel} ...", flush=True)
        try:
            r = subprocess.run([sys.executable, py], cwd=REPO, env=env,
                               capture_output=True, text=True, timeout=TIMEOUT)
            out = r.stdout.rstrip("\n")
            if r.returncode != 0:
                tail = "\n".join(r.stderr.strip().splitlines()[-15:])
                out += f"\n\n[非零退出码 {r.returncode}，stderr 末尾]\n{tail}"
                status = f"❌ exit={r.returncode}"
            else:
                status = "✅"
        except subprocess.TimeoutExpired:
            out = f"[超时 >{TIMEOUT}s]"
            status = "⏱ 超时"
        with open(log, "w", encoding="utf-8") as f:
            f.write(out + "\n")
        print(f"    {status}  → {os.path.relpath(log, REPO)}  ({len(out)} 字符)", flush=True)

    print("\n完成。记得 `python scripts/sync_code.py` 把日志同步进章节。")


if __name__ == "__main__":
    main()
