# CLAUDE.md — 《算法工程师的 Infra 入门》（books/infra）开发规范

> ⚠️ **本书是 `H-Jett/AGILearning` 仓库中的一本**（位于 `books/infra/`），不再是独立仓库根。
> 仓库级约定（books.yml、统一构建、提交前缀、站点地址）见根目录 [`CLAUDE.md`](../../CLAUDE.md)。
> **书内的写作纪律以本文件为准**，仓库级文件不覆盖它。
>
> 站点：<https://h-jett.github.io/AGILearning/infra/>。发布走**仓库级** workflow
> `.github/workflows/pages.yml`（构建全部书）；本书自己的校验命令登记在根 `books.yml` 的 `checks`。

> 本文件是这本书的开发约定，Claude Code 会自动加载。人类或 AI 写这本书时**必须遵守**。
> 目的：让这本《算法工程师的 Infra 入门》保持统一、正确、可移植。

## 1. 这是什么

一本"边学边记"的 infra 入门书，面向**算法工程师**，从推理讲到训练。最终会成书，所以：

> **⚠️ 内容必须非常正确。** 结论尽量在真机上跑出数据来验证；具体数字只当示例，讲的是**规律**。

### 1.1 查证纪律（硬要求，写每一章都适用）

> **不要凭记忆写作。** 模型记忆里的"常识"混杂着过时版本、被误传的数字和张冠李戴的结论，
> 而这本书会成书——一处错会被读者当真很久。

1. **写之前先查**：动笔前检索**一手资料**（论文原文、官方文档、框架源码、官方 benchmark），
   而不是凭印象组织内容。真机实验负责验证"规律对不对"，一手资料负责验证"概念与数字对不对"，
   **两者都要做，不能互相替代**。
2. **一手优先，来源分级**：官方文档 / 论文 / 源码 > 权威教材 > 头部技术媒体与会议分享 >
   个人博客与论坛 > 自媒体。低级别来源只能作线索，**不能作唯一依据**。
3. **关键结论交叉验证**：数字、版本号、接口名、算法细节、性能量级，
   至少**两个独立来源**一致才写；只有一个来源时明确标注"仅见于 X"。
4. **出处要落盘**：引用过的资料登记到章节参考条目或统一索引页，给出可点开链接与查阅日期。
5. **查不到就别写**：宁可写"这一点尚未核实"或缩小结论范围，
   **绝不用听起来合理的措辞填补空白**。发现自己在"推测"时，停下来去查，或显式标注为推测。
6. **区分四层信息**：**事实**（可验证）/ **官方规定或约定**（引出处）/
   **工程惯例**（明说是惯例，会变）/ **传闻**（社区说法，标注证据强度：
   有共识 / 有争议 / 缺乏证据 / 明确错误）。
7. **警惕自己的过时知识**：涉及版本号、API、默认值、硬件规格、生态现状时，
   **默认记忆已经过期**，一律现查。

## 2. 教学法（每章固定结构）

用**中文**写。每章遵循「**概念 → 思考题 → 真机代码练习**」：

1. 概念讲解，配真机实测数据（表格 / 图）；
2. 章末**思考题**（数量**按实际需求，可多可少**——简单章少几道、复杂章多几道），紧扣本章，指向答案册（见 §5）；
3. 附录内联本章练习代码（由脚本从 `.py` 同步，见 §4）。

风格：像给同事讲清楚一件事，多用"为什么"，诚实标注反直觉/局限（"诚实说明"）。

## 3. 单一真源 & 可移植 Markdown（核心原则）

- **内容只存一份**：正文 `docs/`、代码 `exercises/*.py`、图数据脚本。网页/PDF 都是它的渲染。
- **可移植 Markdown**：必须在 **GitHub 原生渲染 + 普通 Markdown 阅读器 + MkDocs** 三边都能看：
  - 用**脚注 + 引用块**做注释，**不要**用 MkDocs 专有的 `!!! note` admonition；
  - 术语表锚点、思考题跳转都用**显式 HTML 锚点** `<a id="xxx"></a>`（三边通用）；
  - 代码**物理内联**成 fenced 代码块（不要用 `--8<--`/`{{#include}}` 这类构建期引用，纯阅读器看不到）。

## 4. 目录结构与命名

```
docs/                       # 正文（唯一内容真源）
  index.md                  # 封面 / 大纲 / 进度
  roadmap.md                # 七部分学习路线图（活的大纲，改动记进"调整记录"）
  glossary.md               # 术语表（每词条前有 <a id> 锚点，按主题分组）
  chapters/NN-partname/     # 每部分一个目录
    00-intro.md             # 该部分开篇导读（每部分都要有）
    NN-name.md              # 章节，两位数字前缀
    figures/                # 该部分的数据图 PNG
  qa/NN-name-qa.md          # 思考题答案册（网页可跳转）
exercises/NN-partname/
  NN_name.py                # 练习代码，章节号前缀（与章节一一对应）
  plot_*.py                 # 生成 figures/ 里 PNG 的绘图脚本
scripts/sync_code.py        # 把 .py 内联进章节附录（含 --check）
mkdocs.yml                  # 站点配置
.github/workflows/docs.yml  # GitHub Pages 自动发布（会跑 sync --check + build --strict）
```

命名对应：第 N 章 → `chapters/PART/NN-name.md`、`exercises/PART/NN_name.py`、`qa/NN-name-qa.md`。

## 5. 各类文件的具体约定

**章节 `.md`**：小节用 `## N.x 标题`。术语首次出现加脚注 `[^x]`，脚注体链接到术语表
`[术语](../../glossary.md#anchor)`。章末思考题下写跳转链接：
`> 📖 **参考答案**（想清楚再看）：[Q1](../../qa/NN-name-qa.md#q1) · [Q2](...#q2) · …`（按实际题数列 Q1…QN）。

**答案册 `docs/qa/NN-name-qa.md`**：**只记「问题 + 正确答案」**，不记读者作答。每题前加锚点
`<a id="q1"></a>`、`<a id="q2"></a>`…（按题数 q1…qN）。必须加进 `mkdocs.yml` 的「思考题参考答案」nav（否则 `--strict` 报错）。

**练习 `.py`**：可直接运行、有必要日志。**仓库内一律用相对路径，禁止绝对路径**（会泄漏用户名等个人信息）。
用到模型时走 `MODEL_PATH = os.environ.get("INFRA_MODEL", "Qwen/Qwen3-0.6B")`（默认从 HF 下载 Qwen3-0.6B，
本地跑可 `INFRA_MODEL=/你的/模型路径 python ...` 覆盖）。章节附录用同步标记包裹：
```
<!-- CODE:exercises/PART/NN_name.py START -->
​```python
（由 scripts/sync_code.py 自动填入，勿手改）
​```
<!-- CODE:exercises/PART/NN_name.py END -->
```

**运行输出（必做）**：每个练习都要**真机跑一遍并把输出存下来、在章节里展示**。由
`scripts/run_exercises.py` 跑 `exercises/*/NN_*.py`、把 stdout 存成 `exercises/PART/outputs/NN_name.txt`（代码输出，唯一真源）；章节代码附录之后用 OUTPUT 标记内联该日志（```text 围栏）：
```
<!-- OUTPUT:exercises/PART/outputs/NN_name.txt START -->
​```text
（由 scripts/sync_code.py 自动填入，勿手改）
​```
<!-- OUTPUT:exercises/PART/outputs/NN_name.txt END -->
```
输出随机器有波动属正常；改了练习后重跑 `run_exercises.py` 刷新输出、再 `sync_code.py` 同步。

**术语表**：新术语加到 `glossary.md` 对应主题下，前置 `<a id="slug"></a>`。

## 6. 画图（Mermaid + matplotlib）

> **随时画图**：只要图比文字更清楚，就配图——这是硬要求，别偷懒用纯文字凑。

- **结构/流程/架构/时序图 → Mermaid**：直接写 ```mermaid 代码块。GitHub 原生渲染、MkDocs 也渲染、
  纯文本可 diff。用于：两阶段流程、block 映射、前缀树、请求生命周期、依赖图等。
- **数据图表 → matplotlib PNG**：用实测数字，脚本 `exercises/PART/plot_*.py` 生成到 `chapters/PART/figures/`，
  章节 `![说明](figures/xxx.png)` 引用。规范：
  - **Okabe-Ito 色盲安全配色**（`#0072B2 #E69F00 #D55E00 #009E73 #56B4E9 #CC79A7`），少用颜色、按固定顺序；
  - **白底**（明暗模式都可读）、**英文标签**（避免 matplotlib 缺中文字体出方框）、细线 + 淡网格、直接标注；
  - **不要双 Y 轴**；生成后**打开图看一眼**有没有重叠/溢出再用。

## 7. 新增一章的标准流程（照做）

1. 写 `exercises/PART/NN_name.py`，用 `python scripts/run_exercises.py` **真机跑通并生成日志**、记下真实数字；
2. 需要图就写 `plot_*.py` 生成 PNG，或在章节里写 Mermaid；
3. 写 `docs/chapters/PART/NN-name.md`（概念 + 真机数据/图 + 若干思考题 + 附录：代码标记 + **运行输出标记**）；
4. 写 `docs/qa/NN-name-qa.md`（问题 + 正确答案，q1/q2/q3 锚点）；
5. 新术语加进 `glossary.md`；
6. 更新 `mkdocs.yml`（章节 nav + 答案 nav）、`docs/index.md`（进度）、`docs/roadmap.md`（状态）；
7. **同步 + 校验 + 严格构建**（见 §8）全绿后再提交。

## 8. 构建与校验（提交前必做）

```bash
python scripts/run_exercises.py        # 跑所有练习，捕获输出到 exercises/*/outputs/（改了练习后重跑；可传编号只跑部分）
python scripts/sync_code.py            # 把 .py 源码 + .txt 运行输出内联进章节
python scripts/sync_code.py --check    # 校验已同步（不一致退出码 1）
mkdocs build --strict                  # 坏链接/警告即失败（CI 也跑这条）
```

三条全过才提交。`--strict` 失败会导致 GitHub Actions 部署失败、网页冻结——务必本地先过。

## 9. 提交与发布

- 作者身份用仓库自己的 `git config user.name/email`（本仓库为 `H-Jett`）。
- 提交信息中文、说清做了什么。
- push 到 `main` 触发 Actions 自动构建并发布到 GitHub Pages。
- `mkdocs.yml` 配 `site_url` + `repo_url` + `repo_name`（网页右上角显示 GitHub 仓库链接）；
  `README.md` 顶部放**在线阅读**链接（GitHub Pages 站点）——网页与仓库互相可达。

## 10. 通用约束（来自用户全局规范）

- **不主动写报告文件**（`*.md` 总结、`report_*.json` 等）；结论直接说，除非明确要求导出。
- 需要时间戳用**北京时间**：`TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M'`。
- **仓库内所有文件禁止出现绝对路径**（尤其带用户名或机器目录布局的）——公开仓库会泄漏个人信息；
  只用相对路径 / 环境变量。聊天里为方便定位可给绝对路径，但**绝不提交进仓库**。
- 大的中间产物放持久大盘、别写爆 `/tmp`（具体机器路径不写进仓库）。
