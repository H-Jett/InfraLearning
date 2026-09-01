#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 10 章练习：探测 GPU 架构，实测峰值算力 / 显存带宽，算出 roofline 拐点。

三件事：
  A) 打印这块卡的硬件参数（SM 数、显存等）。
  B) 峰值算力：同一个大矩阵乘，分别用 fp32(CUDA core) / TF32 / bf16(Tensor Core)，
     实测 TFLOPS —— 亲眼看见 Tensor Core 比普通 CUDA core 快多少。
  C) 显存带宽：大张量逐元素相加，实测 GB/s。
  由 B、C 算出 roofline 拐点（临界算术强度 = 峰值算力 / 带宽），呼应第 1 章 memory/compute-bound。
"""

import time
import torch

DEVICE = "cuda:0"


def sync():
    torch.cuda.synchronize()


def timed(fn, warmup=3, repeat=10):
    for _ in range(warmup):
        fn()
    sync(); t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    sync()
    return (time.perf_counter() - t0) / repeat


def main():
    p = torch.cuda.get_device_properties(DEVICE)
    print("=" * 60)
    print("A) GPU 硬件参数")
    print("=" * 60)
    print(f"名称                : {p.name}")
    print(f"SM 数 (multi_processor_count) : {p.multi_processor_count}")
    print(f"显存                : {p.total_memory/2**30:.1f} GiB")
    print(f"算力架构 (compute cap)        : {p.major}.{p.minor}")

    # ---------- B) 峰值算力：fp32 vs TF32 vs bf16 ----------
    print("\n" + "=" * 60)
    print("B) 峰值算力：大矩阵乘 (N=8192)，不同精度实测 TFLOPS")
    print("=" * 60)
    N = 8192
    flops = 2 * N ** 3  # 一次 N×N×N matmul 的浮点运算数
    a32 = torch.randn(N, N, device=DEVICE, dtype=torch.float32)
    b32 = torch.randn(N, N, device=DEVICE, dtype=torch.float32)
    a16 = a32.to(torch.bfloat16); b16 = b32.to(torch.bfloat16)

    # fp32：关掉 TF32，走真正的 FP32 CUDA core
    torch.backends.cuda.matmul.allow_tf32 = False
    t = timed(lambda: torch.mm(a32, b32))
    print(f"fp32  (CUDA core)   : {flops/t/1e12:>7.1f} TFLOPS   ({t*1e3:.1f} ms)")

    # TF32：开 TF32，fp32 输入走 Tensor Core 的 TF32 通路
    torch.backends.cuda.matmul.allow_tf32 = True
    t = timed(lambda: torch.mm(a32, b32))
    print(f"TF32  (Tensor Core) : {flops/t/1e12:>7.1f} TFLOPS   ({t*1e3:.1f} ms)")

    # bf16：走 Tensor Core
    t = timed(lambda: torch.mm(a16, b16))
    bf16_tflops = flops / t / 1e12
    print(f"bf16  (Tensor Core) : {bf16_tflops:>7.1f} TFLOPS   ({t*1e3:.1f} ms)")

    # ---------- C) 显存带宽 ----------
    print("\n" + "=" * 60)
    print("C) 显存带宽：大张量逐元素相加 (c = a + b)")
    print("=" * 60)
    M = 1 << 26  # 64M 个 float32
    x = torch.randn(M, device=DEVICE)
    y = torch.randn(M, device=DEVICE)
    t = timed(lambda: torch.add(x, y))
    bytes_moved = 3 * M * 4  # 读 a、读 b、写 c，各 4 字节
    bw = bytes_moved / t / 1e12  # TB/s
    print(f"实测带宽            : {bw*1e3:>7.0f} GB/s   ({t*1e3:.2f} ms)")

    # ---------- roofline 拐点 ----------
    print("\n" + "=" * 60)
    print("Roofline 拐点（临界算术强度）")
    print("=" * 60)
    ridge = bf16_tflops * 1e12 / (bw * 1e12)  # FLOP/byte
    print(f"临界算术强度 = 峰值算力 / 带宽 ≈ {ridge:.0f} FLOP/byte")
    print(f"→ 算术强度 < {ridge:.0f}：memory-bound（如 decode，强度≈1）")
    print(f"→ 算术强度 > {ridge:.0f}：compute-bound（如大矩阵乘 prefill）")
    print("这就是第 1 章 memory/compute-bound 判据的定量版本。")


if __name__ == "__main__":
    main()
