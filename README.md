# 算法工程师的 Infra 入门

一本"边学边记"的 infra 入门书。内容以标准 Markdown 编写，**三种方式都能看**：

1. **直接在 GitHub 上浏览** [`docs/`](docs/) 里的 `.md`（脚注、代码、术语表跳转都可用）；
2. **本地用任意 Markdown 阅读器**（Typora / Obsidian / VSCode）打开 `docs/` 里的文件；
3. **构建成网站**（GitHub Pages），带全文搜索和侧边栏导航。

从 [`docs/index.md`](docs/index.md) 开始读。

## 目录结构

```
infra-learning/
  docs/                    # 书的正文（唯一内容真源）
    index.md               # 书的封面 / 大纲 / 进度
    glossary.md            # 术语表（全称 + 解释，显式 <a id> 锚点）
    chapters/01-inference/ # 第一部分 · 推理
  exercises/               # 可运行的练习代码（.py 是代码唯一真源）
  scripts/sync_code.py     # 把 .py 内联进章节附录（保证阅读器里也能看到代码）
  docs/qa/                 # 思考题参考答案（答案册，网页可跳转）
  mkdocs.yml               # 网站构建配置
  .github/workflows/       # GitHub Pages 自动发布
```

**设计原则**：内容只存一份（`docs/*.md` + `exercises/*.py`），网页只是它的另一种渲染。
代码物理内联进 `.md`（由 `sync_code.py` 从 `.py` 自动同步），所以在任何 Markdown 阅读器里都能看到。

## 本地预览网站

```bash
python -m venv .venv && source .venv/bin/activate
pip install mkdocs-material
mkdocs serve            # 打开 http://127.0.0.1:8000
```

## 改了练习代码后

```bash
python scripts/sync_code.py          # 把 .py 内容同步进章节附录
python scripts/sync_code.py --check  # 校验是否已同步（CI 会跑）
```

## 发布到 GitHub Pages

推到 GitHub 后，在仓库 **Settings → Pages → Build and deployment → Source** 选
**GitHub Actions**。之后每次 push 到 `main`，`.github/workflows/docs.yml` 会自动构建并发布。
