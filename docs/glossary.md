# 术语表

> 按主题分组。每个词条给出**全称 + 一句话解释 + 出现章节**。
> 章节里术语首次出现处有脚注，脚注会跳到这里。

## 视觉表征基础

<a id="patchify"></a>
**Patchify（图像分块）** — 把图像切成固定大小（如 16×16 像素）的不重叠小块，每块作为一个 token。
是"图像的 tokenizer"。→ 第 1 章

<a id="patch-embedding"></a>
**Patch Embedding（分块嵌入）** — 把每个 patch 展平后线性投影到模型隐藏维度的那一层。
实现上等价于 `nn.Conv2d(3, d_model, kernel_size=P, stride=P)`。→ 第 1 章

<a id="vision-token"></a>
**视觉 token（Vision Token）** — 图像经视觉编码器和连接器后，送进 LLM 的那些向量。
与文本 token 同形状、同待遇（一样占 KV cache、一样过注意力），但数量由**分辨率**决定。→ 第 1 章

<a id="vit"></a>
**ViT (Vision Transformer)** — 把 Transformer 直接作用在 patch 序列上的视觉模型
（Dosovitskiy et al., 2020）。内部是**双向注意力**，没有因果掩码。→ 第 1、2 章

<a id="token-merge"></a>
**Token Merge（token 合并）** — 把相邻的若干视觉 token 合并成一个，直接减少送入 LLM 的 token 数。
如 Qwen2-VL 系列的 2×2 merge，token 数降到 1/4。→ 第 1、8、11 章

<a id="smart-resize"></a>
**smart_resize** — Qwen-VL 系列 processor 的预处理策略：保持长宽比，把宽高对齐到
`patch_size × merge_size`（28）的倍数，并把总像素数约束在 `[min_pixels, max_pixels]` 内。→ 第 1、8 章

## 模型与架构

<a id="vlm"></a>
**VLM (Vision-Language Model，视觉语言模型)** — 能同时接受图像与文本输入的模型。
典型结构是"视觉编码器 + 连接器 + LLM"。→ 全书

<a id="connector"></a>
**连接器（Connector / Projector）** — 把视觉编码器的输出映射成 LLM 能吃的 token 的模块。
三条主流路线：线性/MLP、Q-Former、cross-attention。→ 第 6 章

## 推理与工程

<a id="kv-cache"></a>
**KV Cache（键值缓存）** — 缓存历史 token 的 Key/Value，避免解码时重复计算。
显存占用 = `2 × 层数 × KV头数 × head_dim × 序列长 × batch × 每元素字节`。
视觉 token 一样要占。→ 第 1、23 章；完整推导见姊妹篇《算法工程师的 Infra 入门》第 2 章

<a id="prefill"></a>
**Prefill（预填充）** — 推理的第一阶段：把整个输入序列一次性前向、填好 KV cache。
VLM 里视觉 token 让这一阶段显著变重。→ 第 1、23 章
