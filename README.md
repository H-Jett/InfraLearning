# 多模态入门：给 LLM 工程师的一本书

一本"边学边记"的多模态入门书，写给**熟悉纯文本 LLM、想转多模态**的算法工程师。
内容以标准 Markdown 编写，**三种方式都能看**：

1. **直接在 GitHub 上浏览** [`docs/`](docs/) 里的 `.md`（脚注、代码、术语表跳转都可用）；
2. **本地用任意 Markdown 阅读器**（Typora / Obsidian / VSCode）打开 `docs/` 里的文件；
3. **构建成网站**（GitHub Pages），带全文搜索和侧边栏导航。

从 [`docs/index.md`](docs/index.md) 开始读；课程全貌见 [`docs/roadmap.md`](docs/roadmap.md)。

## 目录结构

```
multimodal-learning/
  docs/                        # 书的正文（唯一内容真源）
    index.md                   # 封面 / 大纲 / 进度
    roadmap.md                 # 七部分学习路线图
    glossary.md                # 术语表（全称 + 解释，显式 <a id> 锚点）
    chapters/01-vision-basics/ # 第一部分 · 视觉表征基础
    qa/                        # 思考题参考答案
  exercises/                   # 可运行的练习代码（.py 是代码唯一真源）
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
[《算法工程师的 Infra 入门》](https://github.com/H-Jett/InfraLearning)。
