# 第 7 章 投机解码：Speculative Decoding

> 本章目标：讲清投机解码为什么能在**不掉精度**的前提下加速 decode——它是少数能绕开
> "TPOT 下限"那堵墙的技巧。数据用 Qwen3-0.6B 真机实测。

## 7.1 问题：decode 又串行、又访存受限

回顾前几章：decode 一次只出 1 个 token（串行），且是访存受限——每步把整套权重从显存搬一遍，
GPU 算力大量闲着。TPOT 下限 = 模型字节/带宽 是一堵墙：只要还是"一次一个 token"，就撞不破。

但注意一个反差：**GPU 的算力是空闲的**。能不能用这些空闲算力，一次多产出几个 token？

## 7.2 关键 enabler：一次验证 K 个 token 几乎免费

先看一个真机事实。给定一段前缀缓存，让目标模型一次前向处理 K 个 token：

| K（一次处理的 token 数） | 一次前向 | 摊到每 token |
|----------------------:|--------:|-----------:|
| 1  | 16.78 ms | 16.78 ms |
| 2  | 20.41 ms | 10.21 ms |
| 4  | 20.42 ms | 5.10 ms |
| 8  | 19.83 ms | 2.48 ms |
| 16 | 19.71 ms | 1.23 ms |

**一次前向的总耗时几乎不随 K 变（平线）——验证 16 个 token ≈ 验证 1 个。**
这和第 3 章 batching 是同一个道理：权重搬一次的成本，可以摊给更多 token（这次是沿**序列**方向摊）。

于是有了大胆的想法：**先想办法"猜"出接下来 K 个 token，再让目标模型一次性并行验证它们——
反正多验几个几乎免费。**

## 7.3 投机解码的机制

**Speculative Decoding**[^spec] 每一轮做三件事：

1. **草稿 (draft)**：用一个**又小又快**的草稿模型[^draft]，自回归地猜出接下来 K 个 token（便宜）；
2. **验证 (verify)**：目标（大）模型对这 K 个 token 做**一次并行前向**，得到每个位置它自己会给的概率；
3. **接受 / 拒绝**：从头逐个比对，**接受草稿与目标一致的最长前缀**，第一个不一致处由目标重采样一个，
   后面的草稿丢弃。

于是**一次目标前向，能吐出"接受的若干个 + 1 个"token**，而不是死板的 1 个。目标前向的次数少了，
decode 就快了——而目标前向本来就是 memory-bound，多验几个几乎不加钱。

## 7.4 为什么是"无损"的

这是投机解码最漂亮的地方：**输出分布严格等于目标模型自己解码的分布**，一个字都不带偏。

靠的是一套**接受/拒绝采样**：草稿给某 token 的概率是 q、目标是 p，就以 min(1, p/q) 的概率接受；
一旦拒绝，就从修正后的分布重采样。数学上可以证明，这样得到的序列分布和"目标模型自己逐 token 采样"
完全一致。所以投机解码**只加速、不改变结果质量**——这是它和量化、剪枝等"有损"手段的根本区别。

## 7.5 加速多少？取决于接受率

每次目标前向平均能吐出多少 token，取决于**接受率 α**（草稿有多像目标）和草稿长度 K：

> E[tokens] = (1 − α^(K+1)) / (1 − α)　（含 1 个 bonus token）

| 接受率 α | K=2 | K=4 | K=8 |
|-------:|----:|----:|----:|
| 0.3 | 1.39 | 1.43 | 1.43 |
| 0.5 | 1.75 | 1.94 | 2.00 |
| 0.7 | 2.19 | 2.77 | 3.20 |
| 0.9 | 2.71 | 4.10 | 6.13 |

标准解码每次目标前向只吐 1 个 token；投机解码吐 E 个 → **理想加速 ≈ E 倍**（还要扣掉草稿本身的开销）。

- **接受率越高**（草稿越像目标），加速越大；α=0.9、K=8 时一次能吐 6 个 token。
- **接受率太低**时几乎不加速，甚至因草稿开销**反而变慢**——所以草稿模型要选得好。

## 7.6 常见变体

不想额外养一个草稿模型？有几种"自己给自己当草稿"的做法：

- **Medusa**[^medusa]：给模型加几个额外的"预测头"，一次并行预测未来好几个 token 当草稿；
- **EAGLE**：在**特征层**加一个轻量自回归头来产草稿，接受率更高，是目前很强的方案；
- **Prompt lookup / n-gram**：直接从 prompt 或已生成文本里"抄"可能重复的片段当草稿（对代码、长文档改写等重复多的场景特别有效，零额外模型）。

## 7.7 代价与限制

- **接受率看场景**：草稿和目标越接近、任务越"可预测"（代码、结构化文本）接受率越高，收益越大；
- **草稿有开销**：草稿模型的前向也要时间，接受率不够高就得不偿失；
- **高并发时收益缩水**（重要）：投机解码的红利来自"目标模型验证几乎免费"，而这只在目标 **memory-bound**
  时成立。**高 batch 下目标模型已经 compute-bound（GPU 被喂饱，见第 3 章）**，多验 K 个 token 不再免费，
  投机解码的加速就大幅缩水。所以它主要用在**低延迟、低并发**场景（如单用户交互），
  而高吞吐离线场景更多靠 batching。

> 一句话：**投机解码用"空闲算力 + 一次并行验证"换 decode 步数，低并发时是免费的午餐，高并发时午餐要收费。**

## 7.8 思考题

1. 实验 A 里，验证 16 个 token 只比验证 1 个多约 3 ms。为什么正是这个事实让"先猜后验"变得划算？
2. 投机解码号称"无损"（不掉精度）。它靠什么保证最终输出分布和目标模型自己解码完全一致？
3. 为什么投机解码在高并发（大 batch）场景收益会大幅缩水？（提示：回到第 1、3 章，"免费验证"的前提是什么）

> 参考答案见仓库 `qa/07-speculative-decoding-qa.md`。

## 7.9 延伸阅读

- 第 8 章 服务指标：投机解码主要改善单请求延迟（TPOT），怎么和吞吐一起权衡。
- 第 9 章 引擎实战：vLLM / SGLang 都内置了投机解码 / EAGLE 支持。
- 第二部分 · 算子：一次验证多 token 的高效 attention kernel（含 tree attention）是其底层。

## 7.10 附录：本章练习代码

由 `scripts/sync_code.py` 从源码自动同步。源码：`exercises/01-inference/07_speculative_decoding.py`。

<!-- CODE:exercises/01-inference/07_speculative_decoding.py START -->
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 7 章练习：理解投机解码 (speculative decoding) 为什么能加速。

核心 enabler：大模型"一次并行验证 K 个 token"几乎和"解码 1 个 token"一样便宜——
因为 decode 是访存/发射受限，一次前向的成本主要是把权重搬一遍，多验几个 token 几乎免费
（就像 batching，只不过这次是沿"序列长度"方向）。

实验 A（真机）：给定一段前缀缓存，测目标模型一次前向处理 query_len = 1,2,4,8,16 个 token 的耗时。
    预期：基本是平线——验证 8 个 token ≈ 验证 1 个。这正是投机解码"免费验证"的基础。

实验 B（模拟）：给定接受率 α 和草稿长度 K，算"每次目标前向平均吐出多少 token"，
    从而估算理想加速比。

绝对路径：
  /volume/data/hjiang02/workspace/infra-learning/exercises/01-inference/07_speculative_decoding.py
"""

import time
import statistics
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/volume/data/models/Qwen3-0.6B"
DEVICE = "cuda:0"


def sync():
    torch.cuda.synchronize()


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()

    # ---------- 实验 A：一次验证 K 个 token 的耗时 ----------
    print("=" * 64)
    print("实验 A（真机）：目标模型一次前向处理 K 个 token 的耗时")
    print("=" * 64)
    prefix = torch.randint(0, tok.vocab_size, (1, 512), device=DEVICE)

    @torch.no_grad()
    def verify_time(k):
        past = model(prefix, use_cache=True).past_key_values  # 前缀缓存，不计时
        q = torch.randint(0, tok.vocab_size, (1, k), device=DEVICE)
        sync(); t0 = time.perf_counter()
        model(q, past_key_values=past, use_cache=True)
        sync(); return time.perf_counter() - t0

    with torch.no_grad():
        verify_time(1)  # 预热
        print(f"{'K (验证token数)':>16} | {'一次前向(ms)':>14} | {'摊到每token(ms)':>16}")
        print("-" * 54)
        for k in [1, 2, 4, 8, 16]:
            ms = statistics.median([verify_time(k) for _ in range(5)]) * 1000
            print(f"{k:>16} | {ms:>14.2f} | {ms/k:>16.2f}")
    print("\n看：一次前向的总耗时几乎不随 K 变（平线）——验证 8 个 token ≈ 验证 1 个。")
    print("这就是投机解码的基础：既然多验几个几乎免费，那就'先猜后验'，一次验证收获多个 token。")

    # ---------- 实验 B：接受率 → 每次目标前向吐出的 token 数 ----------
    print("\n" + "=" * 64)
    print("实验 B（模拟）：接受率 α、草稿长度 K → 每次目标前向平均吐出多少 token")
    print("=" * 64)
    print("公式：E[tokens] = (1 - α^(K+1)) / (1 - α)   （含 1 个 bonus token）")
    print("标准解码每次目标前向只吐 1 个 token；投机解码吐 E 个 → 理想加速 ≈ E 倍（未扣草稿开销）\n")
    print(f"{'接受率 α':>10} | " + " | ".join(f"K={k:<2}" for k in [2, 4, 8]))
    print("-" * 44)
    for alpha in [0.3, 0.5, 0.7, 0.8, 0.9]:
        row = []
        for K in [2, 4, 8]:
            E = (1 - alpha ** (K + 1)) / (1 - alpha)
            row.append(f"{E:>4.2f}")
        print(f"{alpha:>10.1f} | " + " |  ".join(row))
    print("\n要点：")
    print("  - 接受率 α 越高（草稿越像目标），每次验证吐出的 token 越多，加速越大；")
    print("  - α 太低时几乎不加速，甚至因草稿开销变慢；")
    print("  - 关键：无论接受多少，最终分布严格等于目标模型 → 投机解码是【无损】的（不掉精度）。")
    print("  - 注意：高 batch 时目标模型已 compute-bound，'免费验证'不再免费，投机解码收益变小")
    print("    （所以它主要用在低延迟、低并发场景）。")


if __name__ == "__main__":
    main()
```
<!-- CODE:exercises/01-inference/07_speculative_decoding.py END -->

---

[^spec]: Speculative Decoding（投机解码）：用小模型猜、大模型并行验证，无损加速 decode。详见[术语表](../../glossary.md#speculative-decoding)。
[^draft]: Draft Model（草稿模型）：又小又快、用来提出候选 token 的模型。详见[术语表](../../glossary.md#draft-model)。
[^medusa]: Medusa：给模型加多个预测头做自我投机的方案。详见[术语表](../../glossary.md#medusa)。
