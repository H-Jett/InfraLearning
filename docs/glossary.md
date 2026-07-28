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

<a id="cls-token"></a>
**CLS token（分类 token）** — 拼在 patch 序列最前面的一个**可学习向量**，不来自任何像素；
跑完 encoder 后取它的输出当整图表示。本质是"可学习 query 的加权池化"。
注意 **VLM 通常不用它**（要的是全部 patch token）。→ 第 2 章

<a id="pos-embed"></a>
**位置编码（Position Embedding）** — 给每个 token 注入"我在哪"的向量。ViT 用的是**一维可学习表**
（`[1+N, d]`），**按数量**定义，所以换输入分辨率必须插值、且会让特征漂移。
可外推的替代方案见 2D RoPE / M-RoPE。→ 第 2、8、9 章

<a id="attn-distance"></a>
**平均注意力距离（Mean Attention Distance）** — 用注意力权重加权 query 到各 key 的空间距离，
衡量某个头"看得多远"。实测规律是**越深，局部的头越少**（而不是"底层一定局部"）；
跨模型比较前要按网格尺度归一化。→ 第 2 章

<a id="attn-pooling"></a>
**注意力池化（Attention Pooling）** — 用一个可学习 query 对所有 patch token 做一次
cross-attention 得到整图表示（SigLIP、CoCa 用），可看作把 CLS 的思路独立出来。→ 第 2、4 章

<a id="token-merge"></a>
**Token Merge（token 合并）** — 把相邻的若干视觉 token 合并成一个，直接减少送入 LLM 的 token 数。
如 Qwen2-VL 系列的 2×2 merge，token 数降到 1/4。→ 第 1、8、11 章

<a id="smart-resize"></a>
**smart_resize** — Qwen-VL 系列 processor 的预处理策略：保持长宽比，把宽高对齐到
`patch_size × merge_size`（28）的倍数，并把总像素数约束在 `[min_pixels, max_pixels]` 内。→ 第 1、8 章

## 图文对齐

<a id="clip"></a>
**CLIP (Contrastive Language-Image Pre-training)** — 用对比学习把图像与文本对齐到同一空间的**双塔**模型
（Radford et al., 2021）。两塔独立编码，只在最后算点积，因此可离线建索引、但**没有跨模态细粒度交互**。→ 第 3 章

<a id="infonce"></a>
**InfoNCE** — 对比学习损失：把"找出配对样本"变成 batch 内的 N 选 1 分类，
**batch 内其他样本即负样本**，所以 batch size = 负样本数。CLIP 用它的对称版（图→文 + 文→图）。→ 第 3 章

<a id="temperature"></a>
**温度系数 τ（logit_scale）** — 相似度除以 τ 后再 softmax，控制分布锐度：
τ 越小越尖锐、越放大对难负样本的惩罚。CLIP 把它做成可学习参数，收敛到 **τ≈0.01**（scale=100），
导致 zero-shot 概率**严重饱和且未经校准**——不能当置信度用。→ 第 3 章

<a id="zero-shot"></a>
**zero-shot 分类** — 把类别名写成句子编码成文本向量，看图像向量与谁最近；类别集合可现场更换。
效果依赖 prompt 写法（把输入拉回训练分布），官方用 80 模板集成。→ 第 3 章

<a id="bag-of-words"></a>
**词袋式对齐（Bag-of-Words Behavior）** — 对比模型只优化整句 ↔ 整图的相似度，
对词序、数量、属性绑定不敏感的现象（Winoground、ARO 等基准量化了这一点）。
这是 CLIP 的**结构性**上限，也是 VLM 必须接 LLM 做细粒度推理的原因。→ 第 3、15 章

<a id="penultimate-layer"></a>
**倒数第二层特征（Penultimate Layer）** — LLaVA 系 VLM 取视觉塔**倒数第二层**的 patch token
（而非最后一层或投影后的 pooled 向量）：最后一层为对比目标特化、局部细节被压缩。
实测两层 patch token 平均 cosine 仅 0.62。属工程经验，非定理。→ 第 3、6 章

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
