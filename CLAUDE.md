# CLAUDE.md — MultiModalLearning 开发规范

> 本文件是这个仓库的开发约定，Claude Code 会自动加载。人类或 AI 在此仓库开发时**必须遵守**。
> 目的：让这本《多模态入门：给 LLM 工程师的一本书》保持统一、正确、可移植。

## 1. 这是什么

一本"边学边记"的多模态入门书，读者画像：**熟悉纯文本 LLM、零多模态基础的算法工程师**。
最终会成书，所以：

> **⚠️ 内容必须非常正确。** 结论尽量在真机上跑出数据来验证；具体数字只当示例，讲的是**规律**。

**读者已知**：Transformer / 注意力 / tokenizer / KV cache / 预训练与 SFT / 分布式常识——**不要重复讲**。
**讲法**：处处**从文本世界做类比**再讲差异（"patchify 就是图像的 tokenizer"）。
通用 infra 概念引用姊妹篇《算法工程师的 Infra 入门》（github.com/H-Jett/InfraLearning），不重复造轮子。

## 2. 教学法（每章固定结构）

用**中文**写。每章遵循「**概念 → 思考题 → 真机代码练习**」：

1. 概念讲解，配真机实测数据（表格 / 图）；
2. 章末 **3 道思考题**，紧扣本章，指向答案册（见 §5）；
3. 附录内联本章练习代码 + 运行输出（由脚本从 `.py` / `.txt` 同步，见 §5）。

风格：像给同事讲清楚一件事，多用"为什么"，诚实标注反直觉/局限（"诚实说明"）。

## 3. 单一真源 & 可移植 Markdown（核心原则）

- **内容只存一份**：正文 `docs/`、代码 `exercises/*.py`、图数据脚本。网页/PDF 都是它的渲染。
- **可移植 Markdown**：必须在 **GitHub 原生渲染 + 普通 Markdown 阅读器 + MkDocs** 三边都能看：
  - 用**脚注 + 引用块**做注释，**不要**用 MkDocs 专有的 `!!! note` admonition；
  - 术语表锚点、思考题跳转都用**显式 HTML 锚点** `<a id="xxx"></a>`（三边通用）；
  - 代码**物理内联**成 fenced 代码块（不要用 `--8<--`/`{{#include}}` 这类构建期引用）。

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
  outputs/NN_name.txt       # 练习的真机运行输出（由 run_exercises.py 生成）
  common.py                 # 公共工具（示例图片准备等）
scripts/sync_code.py        # 把 .py 源码 + .txt 输出内联进章节附录（含 --check）
scripts/run_exercises.py    # 真机跑练习并捕获输出
mkdocs.yml                  # 站点配置
.github/workflows/docs.yml  # GitHub Pages 自动发布（会跑 sync --check + build --strict）
```

命名对应：第 N 章 → `chapters/PART/NN-name.md`、`exercises/PART/NN_name.py`、`qa/NN-name-qa.md`。

## 5. 各类文件的具体约定

**章节 `.md`**：小节用 `## N.x 标题`。术语首次出现加脚注 `[^x]`，脚注体链接到术语表
`[术语](../../glossary.md#anchor)`。章末思考题下写跳转链接：
`> 📖 **参考答案**（想清楚再看）：[Q1](../../qa/NN-name-qa.md#q1) · [Q2](...#q2) · [Q3](...#q3)`。

**答案册 `docs/qa/NN-name-qa.md`**：**只记「问题 + 正确答案」**，不记读者作答。每题前加 `<a id="q1"></a>`。
必须加进 `mkdocs.yml` 的「思考题参考答案」nav（否则 `--strict` 报错）。

**练习 `.py`**：可直接运行、有必要日志。**仓库内一律用相对路径，禁止绝对路径**（会泄漏用户名等个人信息）。
模型走环境变量，默认 HF id：

```python
VLM_MODEL  = os.environ.get("MM_VLM",  "Qwen/Qwen2.5-VL-3B-Instruct")
CLIP_MODEL = os.environ.get("MM_CLIP", "openai/clip-vit-base-patch32")
```

示例图片统一走 `exercises/common.py` 的 `sample_image()`：下载到 `assets/sample/`（**已 gitignore**，
不入库、避免版权与体积问题），失败时回退到程序生成的合成图，保证练习永远能跑。

**运行输出（必做）**：每个练习都要**真机跑一遍并把输出存下来、在章节里展示**。
`scripts/run_exercises.py` 跑 `exercises/*/NN_*.py`、把 stdout 存成 `exercises/PART/outputs/NN_name.txt`；
章节代码附录之后用 OUTPUT 标记内联（```text 围栏）。标记格式：

```
<!-- CODE:exercises/PART/NN_name.py START -->     …     <!-- CODE:exercises/PART/NN_name.py END -->
<!-- OUTPUT:exercises/PART/outputs/NN_name.txt START --> … <!-- OUTPUT:… END -->
```

输出随机器有波动属正常；改了练习后重跑 `run_exercises.py` 刷新输出、再 `sync_code.py` 同步。

**术语表**：新术语加到 `glossary.md` 对应主题下，前置 `<a id="slug"></a>`。

## 6. 画图（Mermaid + matplotlib）

> **随时画图**：只要图比文字更清楚，就配图——这是硬要求，别偷懒用纯文字凑。
> 多模态尤其需要图：patch 网格、架构框图、注意力可视化。

- **结构/流程/架构/时序图 → Mermaid**：直接写 ```mermaid 代码块（GitHub 与 MkDocs 都渲染）。
- **数据图表 / 可视化 → matplotlib PNG**：用实测数字，脚本 `exercises/PART/plot_*.py` 生成到
  `docs/chapters/PART/figures/`，章节 `![说明](figures/xxx.png)` 引用。规范：
  - **Okabe-Ito 色盲安全配色**（`#0072B2 #E69F00 #D55E00 #009E73 #56B4E9 #CC79A7`）；
  - **白底**（明暗模式都可读）、**英文标签**（避免 matplotlib 缺中文字体出方框）、细线 + 淡网格；
  - **不要双 Y 轴**；生成后**打开图看一眼**有没有重叠/溢出再用。

## 7. 新增一章的标准流程（照做）

1. 写 `exercises/PART/NN_name.py`，用 `python scripts/run_exercises.py NN` **真机跑通**、记下真实数字；
2. 需要图就写 `plot_*.py` 生成 PNG，或在章节里写 Mermaid；
3. 写 `docs/chapters/PART/NN-name.md`（概念 + 真机数据/图 + 3 思考题 + 附录：代码 + 运行输出标记）；
4. 写 `docs/qa/NN-name-qa.md`（问题 + 正确答案，q1/q2/q3 锚点）；
5. 新术语加进 `glossary.md`；
6. 更新 `mkdocs.yml`（章节 nav + 答案 nav）、`docs/index.md`（进度）、`docs/roadmap.md`（状态）；
7. **同步 + 校验 + 严格构建**（见 §8）全绿后再提交。

## 8. 构建与校验（提交前必做）

```bash
python scripts/run_exercises.py        # 跑所有练习，捕获输出（可传编号只跑部分）
python scripts/sync_code.py            # 把 .py 源码 + .txt 运行输出内联进章节
python scripts/sync_code.py --check    # 校验已同步（不一致退出码 1）
mkdocs build --strict                  # 坏链接/警告即失败（CI 也跑这条）
```

三条全过才提交。`--strict` 失败会导致 GitHub Actions 部署失败、网页冻结——务必本地先过。

## 9. 提交与发布

- 作者身份用仓库自己的 `git config user.name/email`。
- 提交信息中文、说清做了什么。
- push 到 `main` 触发 Actions 自动构建并发布到 GitHub Pages。

## 10. 通用约束

- **不主动写报告文件**（`*.md` 总结、`report_*.json` 等）；结论直接说，除非明确要求导出。
- 需要时间戳用**北京时间**：`TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M'`。
- **仓库内所有文件禁止出现绝对路径**（尤其带用户名或机器目录布局的）——公开仓库会泄漏个人信息；
  只用相对路径 / 环境变量。聊天里为方便定位可给绝对路径，但**绝不提交进仓库**。
- 大的中间产物（模型权重、数据集）放持久大盘，别写爆 `/tmp`；具体机器路径不写进仓库。
