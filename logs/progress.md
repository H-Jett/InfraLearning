# 学习进度日志

> 倒序记录：最新的在最上面。

## 2026-07-15

- 搭好项目骨架，确定教学方式（概念 → 提问 → 代码练习）。
- 确定先学**推理与服务**，后学分布式训练。
- 完成宏观地图：训练看吞吐、推理看延迟+成本；两阶段（prefill/decode）、KV cache、continuous batching、PagedAttention 等概念先建立印象。
- **进入第 1 章：推理的两个阶段（Prefill / Decode）**。

### 第 1 章问答（详见 qa/01-prefill-decode-qa.md）

- 提问"为什么 decode 是 memory-bound" → 用算术强度讲清；记住 TPOT 下限 ≈ 模型大小/带宽。
- Q1（batching 为何接近 1×）：**答对** ✅，补了甜点 batch size 的严谨性。
- Q2（16 token 为何比 256 慢）：本是 overhead-bound 区间的噪声，非趋势。

### 文档/网站基建（2026-07-16 搭好）

- 内容迁入 `docs/`；`docs/index.md` 为书首页，`docs/glossary.md` 术语表（显式 `<a id>` 锚点）。
- 讲义并入第 1 章：补充 A（GPU 存储层级与 HBM）、补充 B（CUDA 异步/同步与基准测试），
  已改写为统一风格，修正 3 处（HBM≠低延迟而是带宽/能效、消费卡用 GDDR、参数量随模型变）。
- `scripts/sync_code.py`：`.py` → `.md` 代码内联同步（含 `--check`），保证阅读器里也能看代码。
- `mkdocs.yml` + `.github/workflows/docs.yml`：MkDocs Material，`mkdocs build --strict` 本地已通过。
- 全部用可移植 Markdown（脚注/引用块/内联代码/显式锚点），GitHub 原生浏览 + Pages 双通。

### 用户已跑第 1 章练习

- 机器更空闲：prefill 16–1024 齐平在 ~18–19ms，4096 跳到 54ms；TPOT 18ms、55 tok/s。
  完美验证 overhead-bound 平线，已作为"第二台机器"数据并入正文。

### 待办

- 用户操作：推 GitHub 新仓库 + 开 Pages（见对话中的步骤）。
- 下一步：第 2 章 KV Cache 原理 + 显存计算。
