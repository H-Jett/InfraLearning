#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 11 章练习：亲眼看见"访存合并 (coalescing)"的威力。

同一个 warp 的 32 个线程如果访问**连续**的内存，硬件能把它们合并成一次（或很少几次）
内存事务，带宽拉满；如果访问**分散**的内存，就退化成很多次小事务，带宽暴跌。

我们搬运同样多的数据（都是读 N 个 float），只改**访问顺序**：
  - 顺序 (sequential)：读 x[0], x[1], x[2], ...   —— 完全合并
  - 跨步 (strided)   ：读 x[0], x[P], x[2P], ...   —— 部分分散
  - 随机 (random)    ：读打乱后的下标            —— 完全分散
比较三者的有效带宽。
"""

import time
import torch

DEVICE = "cuda:0"


def sync():
    torch.cuda.synchronize()


def timed(fn, warmup=3, repeat=20):
    for _ in range(warmup):
        fn()
    sync(); t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    sync()
    return (time.perf_counter() - t0) / repeat


def main():
    N = 1 << 26                      # 64M 个 float32
    x = torch.randn(N, device=DEVICE)
    idx_seq = torch.arange(N, device=DEVICE)
    # 跨步：一个与 N 互质的大步长，遍历所有元素但地址跳着走
    stride = 4096 + 1
    idx_strided = (torch.arange(N, device=DEVICE) * stride) % N
    idx_rand = torch.randperm(N, device=DEVICE)

    # 有效带宽：一次 gather 读 N 个 float(x) + 读 N 个 int64(idx) + 写 N 个 float(y)
    bytes_moved = N * 4 + N * 8 + N * 4

    print(f"数据量 N = {N/1e6:.0f}M float32\n")
    print(f"{'访问模式':>10} | {'耗时(ms)':>9} | {'有效带宽(GB/s)':>14} | {'相对顺序':>8}")
    print("-" * 52)
    base = None
    for name, idx in [("顺序", idx_seq), ("跨步", idx_strided), ("随机", idx_rand)]:
        t = timed(lambda: x[idx])
        bw = bytes_moved / t / 1e9
        if base is None:
            base = bw
        print(f"{name:>10} | {t*1e3:>9.2f} | {bw:>14.0f} | {bw/base:>7.2f}x")

    print("\n结论：搬运的数据量一模一样，只是访问顺序不同——")
    print("      顺序访问能被合并成大事务、带宽拉满；随机访问退化成一堆小事务，带宽大幅下降。")
    print("      这就是为什么写 kernel 要让同一 warp 的线程访问连续内存（coalesced access）。")


if __name__ == "__main__":
    main()
