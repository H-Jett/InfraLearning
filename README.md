# 算法工程师的边学边记

> 📖 **在线阅读**：<https://h-jett.github.io/InfraLearning/>

一个方向一个仓库；这个仓库放**面向算法工程师的技术书**。每本书自成体系、独立成站。

## 书目

| 书 | 主题 | 在线阅读 | 目录 |
|---|---|---|---|
| **算法工程师的 Infra 入门** | 推理 → GPU 算子 → 框架 → 分布式训练，配 RTX 5090 真机实验 | [/infra/](https://h-jett.github.io/InfraLearning/infra/) | [`books/infra`](books/infra) |
| **多模态入门** | 从纯文本 LLM 出发，给 LLM 工程师的多模态一本书 | [/multimodal/](https://h-jett.github.io/InfraLearning/multimodal/) | [`books/multimodal`](books/multimodal) |
| **强化学习：从基础到 LLM 后训练** | 从 MDP、策略梯度讲到 RLHF/PPO/GRPO（🌱 初始化中） | [/rl/](https://h-jett.github.io/InfraLearning/rl/) | [`books/rl`](books/rl) |

## 共同的写法

- **教学法**：概念 → 思考题 → 真机代码练习；思考题答案单独成册。
- **正确性纪律**：结论尽量在真机上跑出数据来验证；概念与数字先查**一手资料**（论文 / 官方文档 / 源码），
  关键结论交叉验证；查不到就**宁可不写**或标注为推测。
- **可移植 Markdown**：在 GitHub / 任意 Markdown 阅读器 / MkDocs 三边都能读。

## 仓库结构

```
books.yml                  # 单一真源：有哪些书、站点地址、每本书的配色与校验命令
scripts/build_site.py      # 逐本校验 + mkdocs build → site/<slug>/，再生成索引页与 PWA
scripts/make_icons.py      # 站点图标（本地生成、产物提交；CI 只 --check）
assets/icons/              # 图标产物 + source/（素材与许可）
books/<slug>/              # 一本书 = 一个完整的 MkDocs 工程
  mkdocs.yml  docs/  exercises/  scripts/  CLAUDE.md  README.md
.github/workflows/pages.yml
```

**各书之间完全隔离**：各自的 `mkdocs.yml`、主题、搜索索引、校验脚本、`CLAUDE.md`。
构建脚本只负责按 `books.yml` 逐本调用，不碰书的内部。

## 本地构建

```bash
pip install -r requirements.txt

python scripts/build_site.py                # 全部书（含各书自己的校验）
python scripts/build_site.py --only infra   # 只构建一本
python scripts/build_site.py --check        # 只跑校验不构建
```

产物在 `site/`：`site/index.html` 是书架索引，`site/<slug>/` 是各本书；PWA（manifest + service worker）可加主屏、离线读。

单独调试某一本时，进到书目录直接用 mkdocs 即可（`site_url` 会回落到 localhost）：

```bash
cd books/infra && mkdocs serve
```

## 加一本新书

1. `books/<slug>/` 建一个完整的 MkDocs 工程（可照抄现有任意一本的骨架）；
2. `mkdocs.yml` 里 `site_url` 写成 `!ENV [BOOK_SITE_URL, "http://127.0.0.1:8000/"]`，
   `edit_uri` 写成 `edit/main/books/<slug>/docs/`；
3. 在 `books.yml` 加一条记录（含配色与该书自己的校验命令）。

索引页会自动出现新卡片，不用手工维护列表。

## 姊妹仓库

| 仓库 | 主题 |
|---|---|
| [LifeLearning](https://github.com/H-Jett/LifeLearning) | 边学边记的「生活」方向：系统学茶、珠宝入门 |
