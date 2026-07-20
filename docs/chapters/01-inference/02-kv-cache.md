# 第 2 章 KV Cache：原理与显存计算

> 本章目标：讲清 KV cache 到底缓存了什么、为什么能省、显存怎么一字节字节算出来，
> 以及为什么它是长上下文 / 高并发下的头号显存杀手。数据用 Qwen3-0.6B 真实 config 算并实测。

## 2.1 回顾：为什么需要 KV cache

第 1 章实验 C 里我们见过：关掉 KV cache，每生成一个 token 都要把**整段历史重算一遍**，
生成越长越慢，总计算量是 O(N²)。KV cache 就是为了消掉这个"重算"。要理解它缓存了什么，
得回到注意力的公式。

## 2.2 从注意力公式看：到底缓存了什么

生成第 `t` 个 token 时，这一层的注意力做的是：

```
out_t = softmax( Q_t · [K_1, K_2, ..., K_t]ᵀ / √d ) · [V_1, V_2, ..., V_t]
```

看清楚这里用到了什么：

- **`Q_t`**：只需要当前这个 token 的 Query（刚算出来）。
- **`K_1..K_t` 和 `V_1..V_t`**：需要**从头到现在所有 token** 的 Key 和 Value。

而 `K_1..K_{t-1}`、`V_1..V_{t-1}` 在之前的步骤里**早就算过了**。既然它们不变，
就把它们**存起来**，每步只新算当前 token 的 `K_t`、`V_t`，追加进去即可。这块存储就是 **KV Cache**[^kv]。

> **注：为什么不缓存 Q，也不缓存注意力输出？**
> 因为每个 token 的 `Q` 只在它自己那一步用一次，之后再不用；注意力的输出也不会被后续步骤复用。
> 只有过去的 `K`、`V` 会被后面每一步反复读取——所以只缓存 K 和 V。

## 2.3 KV cache 省的到底是什么

关键要说准确，别笼统说成"O(N²) 降到 O(N)"：

- **不用 cache**：生成第 `t` 个 token，要把**整个模型**（所有层的投影 + FFN）在 `t` 个 token 上
  重跑一遍 → 每步 O(t) 个 token 的完整前向 → 总量 O(N²) 次完整前向。
- **用 cache**：每步只把完整前向跑在**1 个新 token** 上，历史的 K/V 直接从缓存读。
  注意力打分那步 `Q_t · K_{1..t}` 仍是 O(t)（要和 t 个 key 做点积），但这远比"在 t 个 token 上
  重跑 FFN 和投影"便宜。

所以精确的说法是：**KV cache 把"每步在 t 个 token 上重跑整个模型"变成"每步只在 1 个 token 上跑
整个模型 + 读取历史 KV"**。省下的是海量的重复前向计算（代价是拿显存去换）。

## 2.4 显存公式：一字节字节算

KV cache 要为**每一层、每个 KV 头、每个历史 token** 存一份 K 和一份 V。所以：

> **KV_bytes = 2 × 层数 × KV头数 × head_dim × 序列长 × batch × 每元素字节**

逐项拆开：

| 因子 | 含义 |
|------|------|
| `2` | K 和 V 各存一份 |
| `层数` (num_hidden_layers) | 每层都有独立的 KV cache |
| `KV头数` (num_key_value_heads) | **注意是 KV 头，不是 attention 头**——GQA/MQA 会更少 |
| `head_dim` | 每个头的维度 |
| `序列长` (seq_len) | prompt + 已生成的总长度 |
| `batch` | 并发的序列条数 |
| `每元素字节` | bf16/fp16 = 2，fp8 = 1，fp32 = 4 |

> **两个最容易漏 / 错的因子（真实踩坑）：**
>
> 1. **别忘了乘层数**。每一层都有自己**独立**的一份 KV cache，28 层就要 ×28。
>    只算单层、忘了乘层数，会把显存低估几十倍——足以让"卡放得下"的判断变成实际 OOM。
> 2. **是 KV 头，不是 attention 头**。现代模型普遍用 **GQA**[^gqa]（多个 Q 头共享一组 KV 头），
>    KV 头数往往只有 attention 头数的 1/2、1/4 甚至 1/8。用 attention 头数会高估显存。

## 2.5 用 Qwen3-0.6B 真实 config 算，并和实测对上

Qwen3-0.6B 的关键配置：

| 项 | 值 |
|----|----|
| 层数 | 28 |
| attention 头 | 16 |
| **KV 头** | **8**（GQA，比 Q 头少一半） |
| head_dim | 128 |
| 精度 | bf16（2 字节） |

**每个 token 的 KV cache** = 2 × 28 × 8 × 128 × 2 = **114,688 字节 = 112 KiB**。

按公式算不同长度（batch=1），并和真机量出的 `past_key_values` 字节数对比：

| seq_len | 公式 | 真机实测 | 一致 |
|--------:|-----:|---------:|:----:|
| 128     | 14.00 MiB  | 14.00 MiB  | ✅ |
| 1024    | 112.00 MiB | 112.00 MiB | ✅ |
| 4096    | 448.00 MiB | 448.00 MiB | ✅ |
| 40960（满上下文） | 4.38 GiB | — | （公式外推） |

**公式和实测一字节不差**。这说明这条公式不是近似，就是 KV cache 的真实占用。

## 2.6 三个要命的结论

1. **线性增长，双重放大**：KV cache 随 `seq_len` 和 `batch` **线性增长**。上下文翻倍、并发翻倍，
   显存就翻倍。这是高并发长上下文服务最硬的显存约束。

2. **KV cache 会超过模型本身**：满上下文（40960）时，**单条序列**的 KV cache 是 **4.38 GiB**，
   而整个模型权重（596M 参数，bf16）才 **1.11 GiB**——KV cache 是模型的 **3.9 倍**！
   模型越小、上下文越长，这个比例越夸张。

3. **GQA 直接砍显存**：Qwen3-0.6B 用 8 个 KV 头而非 16 个，KV cache 直接**减半**。
   如果用 MQA[^mqa]（只 1 个 KV 头），还能再砍到 1/8。

## 2.7 怎么省 KV cache（后续章节预告）

围绕公式里的每一项，都有对应的省法：

- **减 `KV头数`**：GQA / MQA（模型结构层面）；更进一步是 DeepSeek 的 **MLA**（把 KV 压成低秩潜向量）。
- **减 `每元素字节`**：KV cache 量化（fp8 / int8）——第 6 章。
- **减浪费**：**PagedAttention** 用分页消除显存碎片，让同样的显存装下更多序列——第 4 章。
- **减 `seq_len` 的有效长度**：sliding window attention 等。

## 2.8 思考题

1. batch=32、seq_len=8192 时，Qwen3-0.6B 的 KV cache 有多大？（用公式算）它是模型权重的多少倍？
   一张 32 GiB 的卡光放这份 KV cache 够吗？
2. 为什么缓存的是 K 和 V，而不是 Q？也不是注意力的最终输出？
3. GQA 把 KV 头从 16 减到 8，KV cache 减半——可推理质量为什么不会跟着"减半"？
   （提示：16 个 Q 头一个没少。）

> 参考答案见仓库 `qa/02-kv-cache-qa.md`。

## 2.9 延伸阅读

- 第 4 章 PagedAttention：从"显存碎片"角度进一步榨干 KV cache 的空间。
- 第 6 章 量化：把 KV cache 的每元素字节从 2 砍到 1。
- 第五部分「训练工程」第 31 章：本章的显存计算思路会直接迁移到**训练显存预算**
  （params + grads + optim + activations）。

## 2.10 附录：本章练习代码

由 `scripts/sync_code.py` 从源码自动同步。源码：`exercises/01-inference/kv_cache_memory.py`。

<!-- CODE:exercises/01-inference/kv_cache_memory.py START -->
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 2 章练习：把 KV Cache 的显存"一字节字节"算出来，并和真机实测对上。

我们要验证的公式（单位：字节）：
    KV_bytes = 2 × num_layers × num_kv_heads × head_dim × seq_len × batch × bytes_per_elem
其中 2 = K 和 V 各一份。注意是 num_kv_heads（GQA/MQA 会小于 attention heads）。

三件事：
  A) 用 Qwen3-0.6B 的真实 config 手算 per-token 和不同长度的 KV cache 大小；
  B) 真机跑一遍，量出 past_key_values 的实际字节数，和公式对比；
  C) 对比 GQA vs 若为 MHA 的差异，以及 KV cache 与模型权重的"交叉点"。

绝对路径：
  /volume/data/hjiang02/workspace/infra-learning/exercises/01-inference/kv_cache_memory.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/volume/data/models/Qwen3-0.6B"
DEVICE = "cuda:0"


def kv_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch, bytes_per_elem):
    """KV cache 显存公式。"""
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_elem


def human(n):
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if n < 1024 or unit == "GiB":
            return f"{n:.2f} {unit}"
        n /= 1024


def measure_cache_bytes(past):
    """把 past_key_values 里所有张量的字节数加总（兼容 DynamicCache 和旧式 tuple）。"""
    tensors = []
    if hasattr(past, "key_cache"):          # 较新的 DynamicCache
        tensors = list(past.key_cache) + list(past.value_cache)
    elif hasattr(past, "layers"):           # 更新版本按 layer 组织
        for layer in past.layers:
            tensors += [getattr(layer, "keys", None), getattr(layer, "values", None)]
    else:                                    # 旧式：tuple(tuple(K, V))
        for layer in past:
            tensors += list(layer)
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def main():
    cfg = AutoModelForCausalLM.from_pretrained(MODEL_PATH).config
    L = cfg.num_hidden_layers
    H_attn = cfg.num_attention_heads
    H_kv = cfg.num_key_value_heads
    d = cfg.head_dim
    b = 2  # bf16 = 2 字节
    print("=" * 64)
    print("Qwen3-0.6B 关键 config")
    print("=" * 64)
    print(f"层数 num_hidden_layers      = {L}")
    print(f"注意力头 num_attention_heads = {H_attn}")
    print(f"KV 头   num_key_value_heads = {H_kv}   ← GQA：比 Q 头少！")
    print(f"每头维度 head_dim            = {d}")
    print(f"精度                         = bf16 ({b} 字节)")

    # ---------- A) 手算 ----------
    print("\n" + "=" * 64)
    print("A) 用公式手算 KV cache 大小（batch=1）")
    print("=" * 64)
    per_tok = kv_bytes(L, H_kv, d, 1, 1, b)
    print(f"每个 token 的 KV cache = 2×{L}×{H_kv}×{d}×{b} = {per_tok} 字节 = {human(per_tok)}")
    print(f"\n{'seq_len':>10} | {'KV cache':>14}")
    print("-" * 28)
    for n in [128, 1024, 4096, 40960]:
        print(f"{n:>10} | {human(kv_bytes(L, H_kv, d, n, 1, b)):>14}")

    # ---------- B) 真机实测，和公式对比 ----------
    print("\n" + "=" * 64)
    print("B) 真机实测 past_key_values 字节数 vs 公式")
    print("=" * 64)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()

    print(f"{'seq_len':>10} | {'实测':>14} | {'公式':>14} | 是否一致")
    print("-" * 56)
    with torch.no_grad():
        for n in [128, 1024, 4096]:
            ids = torch.randint(0, tok.vocab_size, (1, n), device=DEVICE)
            out = model(ids, use_cache=True)
            measured = measure_cache_bytes(out.past_key_values)
            formula = kv_bytes(L, H_kv, d, n, 1, b)
            ok = "✅" if measured == formula else "❌"
            print(f"{n:>10} | {human(measured):>14} | {human(formula):>14} | {ok}")

    # ---------- C) GQA vs MHA，以及与模型权重的对比 ----------
    print("\n" + "=" * 64)
    print("C) GQA 省了多少？KV cache 何时超过模型权重？")
    print("=" * 64)
    n = cfg.max_position_embeddings
    kv_gqa = kv_bytes(L, H_kv, d, n, 1, b)
    kv_mha = kv_bytes(L, H_attn, d, n, 1, b)  # 假设不用 GQA，KV 头 = attention 头
    n_params = sum(p.numel() for p in model.parameters())
    model_bytes = n_params * 2  # bf16
    print(f"最大上下文 seq_len = {n}")
    print(f"KV cache（GQA, {H_kv} KV 头）      = {human(kv_gqa)}")
    print(f"KV cache（若 MHA, {H_attn} KV 头）  = {human(kv_mha)}   → GQA 省了 {kv_mha/kv_gqa:.1f}x")
    print(f"模型权重（{n_params/1e6:.0f}M 参数, bf16） = {human(model_bytes)}")
    print(f"\n结论：满上下文时单条序列的 KV cache（{human(kv_gqa)}）已 {kv_gqa/model_bytes:.1f}× 于模型权重本身！")
    print("这就是为什么长上下文 / 高并发下，KV cache 是头号显存杀手。")


if __name__ == "__main__":
    main()
```
<!-- CODE:exercises/01-inference/kv_cache_memory.py END -->

---

[^kv]: KV Cache（键值缓存）：缓存历史 token 的 Key/Value，避免解码时重算。详见[术语表](../../glossary.md#kv-cache)。
[^gqa]: GQA (Grouped-Query Attention，分组查询注意力)：多个 Q 头共享一组 KV 头。详见[术语表](../../glossary.md#gqa)。
[^mqa]: MQA (Multi-Query Attention，多查询注意力)：所有 Q 头共享同一组 KV。详见[术语表](../../glossary.md#mqa)。
