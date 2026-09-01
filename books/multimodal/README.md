# 多模态入门：给 LLM 工程师的一本书

### 📖 在线阅读：<https://h-jett.github.io/MultiModalLearning/>

（带全文搜索、侧边栏导航、明暗主题；每次 push 自动更新）

一本"边学边记"的多模态入门书，写给**熟悉纯文本 LLM、想转多模态**的算法工程师。
内容以标准 Markdown 编写，**三种方式都能看**：

1. **在线网站**：<https://h-jett.github.io/MultiModalLearning/>（推荐）；
2. **直接在 GitHub 上浏览** [`docs/`](docs/) 里的 `.md`（脚注、代码、术语表跳转都可用）；
3. **本地用任意 Markdown 阅读器**（Typora / Obsidian / VSCode）打开 `docs/` 里的文件。

从 [`docs/index.md`](docs/index.md) 开始读；课程全貌见 [`docs/roadmap.md`](docs/roadmap.md)。

## 快速入口

| | 链接 |
|---|---|
| 学习路线图（七部分 39 章 + 5 个实战项目） | [网页](https://h-jett.github.io/MultiModalLearning/roadmap/) · [Markdown](docs/roadmap.md) |
| 第 1 章 · 图像如何变成 token | [网页](https://h-jett.github.io/MultiModalLearning/chapters/01-vision-basics/01-image-to-tokens/) · [Markdown](docs/chapters/01-vision-basics/01-image-to-tokens.md) |
| 🛠 实战项目 1 · 图像 token 预算计算器 | [网页](https://h-jett.github.io/MultiModalLearning/chapters/01-vision-basics/P1-token-budget/) · [Markdown](docs/chapters/01-vision-basics/P1-token-budget.md) |
| 术语表 | [网页](https://h-jett.github.io/MultiModalLearning/glossary/) · [Markdown](docs/glossary.md) |

## 目录结构

```
multimodal-learning/
  docs/                        # 书的正文（唯一内容真源）
    index.md                   # 封面 / 大纲 / 进度
    roadmap.md                 # 七部分学习路线图
    glossary.md                # 术语表（全称 + 解释，显式 <a id> 锚点）
    chapters/01-vision-basics/ # 第一部分 · 视觉表征基础（章节 + 该部分的实战项目题目）
    qa/                        # 思考题参考答案
  exercises/                   # 章节练习代码：我跑给你看（.py 是代码唯一真源）
  projects/                    # 实战项目骨架：你自己写（带 TODO + check.py 自动评分）
  scripts/sync_code.py         # 把 .py 源码 + 运行输出内联进章节附录
  scripts/run_exercises.py     # 真机跑练习、捕获输出
  mkdocs.yml                   # 网站构建配置
  .github/workflows/           # GitHub Pages 自动发布
```

**设计原则**：内容只存一份（`docs/*.md` + `exercises/*.py`），网页只是它的另一种渲染。
代码物理内联进 `.md`（由 `sync_code.py` 自动同步），所以在任何 Markdown 阅读器里都能看到。

## 跑练习

```bash
pip install torch transformers pillow numpy matplotlib
python scripts/run_exercises.py          # 全部；或 python scripts/run_exercises.py 01
```

模型默认从 HuggingFace 拉取，可用环境变量指向本地副本：

```bash
export MM_VLM=/your/path/Qwen2.5-VL-3B-Instruct
export MM_CLIP=/your/path/clip-vit-base-patch32
```

## 本地预览网站

```bash
python -m venv .venv && source .venv/bin/activate
pip install mkdocs-material
mkdocs serve            # 打开 http://127.0.0.1:8000
```

## 改了练习代码后

```bash
python scripts/run_exercises.py NN   # 重跑该练习，刷新输出
python scripts/sync_code.py          # 同步进章节附录
python scripts/sync_code.py --check  # 校验（CI 会跑）
mkdocs build --strict                # 严格构建（CI 会跑）
```

## 姊妹篇

通用推理与训练 infra（KV cache、PagedAttention、CUDA kernel、分布式并行）见
[《算法工程师的 Infra 入门》](https://github.com/H-Jett/AGILearning)。
