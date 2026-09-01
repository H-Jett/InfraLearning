# 多模态入门：给 LLM 工程师的一本书

> 一本"边学边记"的书，写给**已经熟悉纯文本 LLM、想转多模态**的算法工程师。
> 力求**正确**：书中的结论尽量在真机上跑出数据来验证，具体数字只当示例，讲的是规律。

## 这本书的出发点

你已经会的：Transformer、注意力、tokenizer、KV cache、预训练与 SFT。
你缺的其实不多，但很关键——**图像/音频这些连续信号，怎么变成 LLM 能吃的东西**，
以及围绕这件事长出来的一整套架构、训练范式和工程问题。

所以本书的讲法是：**处处从你已知的文本世界出发做类比**，
（"patchify 就是图像的 tokenizer"、"视觉 token 也要占 KV cache"），再讲差异。

## 怎么读这本书

- **正文**讲概念，配**真机实验**（代码在每章附录里，可直接复制运行）。
- 每章末尾有思考题（数量按内容而定），想清楚再看[参考答案](qa/01-image-to-tokens-qa.md)。
- **术语**首次出现处有脚注，全称与解释汇总在[术语表](glossary.md)。
- **每个部分末尾有一个[实战项目](chapters/01-vision-basics/P1-token-budget.md)**（全书 5 个，刻意不密）：
  我给题目 + 骨架 + 自动评分脚本，你自己写代码、跑 `python check.py` 验收。
  这是把"看懂"变成"会做"的关键一步。

## 全书大纲

完整课程见[**学习路线图**](roadmap.md)。分七部分：

| 部分 | 主题 | 你会获得 |
|------|------|----------|
| 一 | 从 LLM 到 VLM：视觉表征基础 | 看懂图像如何变成序列（🔜 进行中） |
| 二 | VLM 架构：把视觉接进 LLM | 读懂任一 VLM 的结构与设计取舍 |
| 三 | 多模态训练：数据与范式 | 设计一套从零到可用的训练配方 |
| 四 | 多模态训练工程与规模化 ★ | 变长 token / 显存 / 并行 / 吞吐全打通 |
| 五 | 多模态推理与部署 | 给 VLM 服务做容量规划与调优 |
| 六 | 生成方向：扩散与统一模型 | 讲清扩散每一步，跑通生成管线 |
| 七 | 音频、全模态与实战 | 语音这条腿 + 端到端实战 |

**当前进度（第一部分）**：

- ✅ 第 1 章 [图像如何变成 token](chapters/01-vision-basics/01-image-to-tokens.md)
- ✅ 第 2 章 [ViT：把 Transformer 用到图像上](chapters/01-vision-basics/02-vit.md)
- ✅ 第 3 章 [CLIP：对比学习怎么把图文对齐](chapters/01-vision-basics/03-clip.md)
- 🛠 [实战项目 1 · 图像 token 预算计算器](chapters/01-vision-basics/P1-token-budget.md)（可做）
- 🔜 第 4 章 视觉编码器家族（CLIP / SigLIP / DINOv2）

## 学习环境

- 硬件：NVIDIA RTX 5090 (32 GiB) 单卡即可跑完第一部分的全部练习
- 软件：torch 2.8 (CUDA) · transformers 4.55 · Pillow 11 · numpy 1.26
- 实验用模型：Qwen2.5-VL-3B-Instruct（读它的 processor/config，够小够典型）、CLIP ViT-B/32

## 姊妹篇

通用推理与训练 infra（KV cache、PagedAttention、CUDA kernel、分布式并行）见
[《算法工程师的 Infra 入门》](https://github.com/H-Jett/InfraLearning)。本书遇到这些概念直接引用，不重复。
