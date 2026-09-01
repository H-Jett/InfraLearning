# CLAUDE.md — 仓库级约定（多本书的单仓库）

> 这是**仓库级**规范。每本书**另有自己的 `books/<slug>/CLAUDE.md`**，
> 在某本书的目录下干活时，那份才是主规范——**书内的写作纪律以书自己的 CLAUDE.md 为准**。
> 本文件只管"多本书共处一个仓库"带来的那些事。

## 1. 这个仓库是什么

一个方向一个仓库。这个仓库是「**面向算法工程师的技术书**」这个方向，站点
<https://h-jett.github.io/InfraLearning/>。

目前两本：`books/infra`（算法工程师的 Infra 入门）、`books/multimodal`（多模态入门）。

> 历史：本仓库原是单本《InfraLearning》，2026-09 起改造成书架仓库——
> 原 infra 内容降到 `books/infra/`，并把姊妹书 MultiModalLearning 并入 `books/multimodal/`。

## 2. 铁律：书与书之间互不干涉

- **不要**把各书的 `mkdocs.yml`、主题配置、校验脚本"统一"或抽成公共模块。
  它们各自演化（infra/multimodal 各有自己的 `scripts/sync_code.py`、练习代码、真机输出）——
  **这些差异是刻意的，不是重复代码**。
- **不要**跨书引用内容（`../multimodal/...`）。要引用就复述并注明"另一本书里讲过"。
- 各书**独立的搜索索引**。搜索不跨书是设计，不是缺陷。

## 3. 单一真源：`books.yml`

**加一本书 = 建目录 + 在 `books.yml` 加一条记录。** 构建脚本据此决定构建哪些、
索引页列哪些，所以不会出现"书加了但索引忘了改"。

每条记录里的 `checks` 是**该书自己的校验命令**（在书目录下执行）。
新书如果有特殊校验，写进它自己的记录，不要塞进 `build_site.py`。

## 4. 站点地址与 `site_url`

- **`books.yml` 的 `base_url` 是唯一写死站点地址的地方**；
  各书 `mkdocs.yml` 里写 `site_url: !ENV [BOOK_SITE_URL, "http://127.0.0.1:8000/"]`，
  由 `scripts/build_site.py` 注入。**仓库改名时只改 `books.yml` 一行。**
- `edit_uri` 必须写成 `edit/main/books/<slug>/docs/`，否则"在 GitHub 上编辑"会指错目录。
- ⚠️ **`github.io` 的路径改名后不会重定向**（实测过，直接 404，
  GitHub 只重定向 `github.com` 上的仓库地址）。所以 `base_url` 和各书 `slug` **一次定好**。

## 5. 配色（每本书一套专属色，构建时查重）

- 每本书在 `books.yml` 里有 `palette.primary/accent`，`build_site.py` 据此生成
  `books/<slug>/docs/assets/palette.css` 与 `overrides/main.html`（**生成物但要提交**）。
- 两本书 primary 的感知距离 **< 120 直接构建失败**（挡住撞色）。加新书想自己挑色可先算：
  ```python
  import sys; sys.path.insert(0, 'scripts')
  from build_site import color_distance, MIN_DISTANCE
  print(color_distance('#你的色', '#0E7490'), MIN_DISTANCE)
  ```
- 现用：infra `#0E7490` 青蓝、multimodal `#B5541F` 赤橙（两两 ≥ 120）。

## 6. 图标（生成物但要提交，CI 只 --check）

`assets/icons/` 是 `scripts/make_icons.py` 本地生成、**提交进仓库**的产物（🧐 叠深色圆角底）。
CI 只跑 `make_icons.py --check` 校验齐备——因为栅格化依赖 `libcairo2`，不想让发布流程多一个系统级依赖。
素材是 Noto Emoji 矢量（Apache-2.0），归档在 `assets/icons/source/` 并有 PROVENANCE.md。

## 7. 构建与校验（提交前必做）

```bash
python scripts/build_site.py            # 全部书：各自校验 + mkdocs build --strict + 索引页 + PWA
python scripts/build_site.py --only infra   # 只动了一本书时
python scripts/build_site.py --check    # 只校验（含生成物是否与 books.yml 同步）
```

任何一本书的校验或构建失败，整个构建就失败——**站点不会发布半成品**。
单独调试某本书时进到书目录用 mkdocs（`site_url` 回落到 localhost，不影响）：
`cd books/infra && mkdocs serve`。

## 8. 提交

- 提交信息中文、说清做了什么 + 为什么；**每完成一块可独立成立的内容就提交**。
- **提交信息里标明动的是哪本书**，例如 `infra: 第 13 章 算子优化基本功` / `multimodal: 修正第 3 章`；
  改仓库级的东西（构建脚本、索引页、books.yml）用 `repo:` 前缀。
- 作者身份用仓库自己的 `git config user.name/email`（本仓库为 `H-Jett`）。
- push 到 `main` 触发 Actions 构建全部书并发布。

## 9. 通用约束（来自用户全局规范）

- **不主动写报告文件**（`*.md` 总结、`report_*.json` 等）；结论直接说，除非明确要求导出。
- 需要时间戳用**北京时间**：`TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M'`。
- **仓库内所有文件禁止出现绝对路径**（尤其带用户名或机器目录布局的）——公开仓库会泄漏个人信息；
  只用相对路径 / 环境变量。聊天里为方便定位可给绝对路径，但**绝不提交进仓库**。
