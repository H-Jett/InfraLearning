#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 12 章练习：用 torch.profiler 找瓶颈——看清"哪个 kernel 在耗时、走的哪条硬件路径"。

我们对同一个矩阵乘用 fp32 和 bf16 两种精度各跑一遍，用 profiler 抓出：
  - 占用 GPU 时间最多的 kernel 是谁（名字里能看出走 CUDA core 还是 Tensor Core）；
  - 每次调用耗时多少。
这正是第 10 章"fp32 用 CUDA core、bf16 用 Tensor Core"的结论——用 profiler 亲手验出来。
"""

import torch
from torch.profiler import profile, ProfilerActivity


def cuda_us(e):
    # 兼容不同 torch 版本的字段名
    return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)


def top_kernel(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    ka = sorted(prof.key_averages(), key=cuda_us, reverse=True)
    for e in ka:
        if cuda_us(e) > 0:
            return e.key, cuda_us(e) / 1e3 / iters  # ms/call
    return "?", 0.0


def main():
    N = 8192
    a32 = torch.randn(N, N, device="cuda"); b32 = torch.randn(N, N, device="cuda")
    a16 = a32.to(torch.bfloat16); b16 = b32.to(torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = False  # 让 fp32 走真正的 CUDA core

    print("用 torch.profiler 抓 8192³ 矩阵乘，最耗时的 GPU kernel：\n")
    for name, fn in [("fp32", lambda: torch.mm(a32, b32)),
                     ("bf16", lambda: torch.mm(a16, b16))]:
        k, ms = top_kernel(fn)
        print(f"[{name}] {ms:6.2f} ms/call")
        print(f"       主 kernel: {k[:72]}\n")

    print("读法：profiler 直接告诉你时间花在哪个 kernel 上；kernel 名里的 simt=CUDA core、")
    print("      含 tensor/wgmma/hmma 等字样=Tensor Core。这样不必猜，一眼看出走了哪条硬件路径。")


if __name__ == "__main__":
    main()
