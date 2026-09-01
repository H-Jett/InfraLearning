# 强化学习：从基础到 LLM 后训练

> 📖 **在线阅读**：<https://h-jett.github.io/AGILearning/rl/>
> 本书是书架仓库 [H-Jett/AGILearning](https://github.com/H-Jett/AGILearning) 中的一本，位于 `books/rl/`。

一本"边学边记"的强化学习入门书，从 MDP、策略梯度一路讲到 RLHF、PPO、GRPO——
面向**熟悉深度学习与大语言模型、但没系统学过 RL** 的算法工程师。

- **教学法**：概念 → 思考题 → 真机代码练习（RTX 5090）；思考题答案单独成册；穿插少量动手实战项目。
- **正确性纪律**：概念与数字先核对**一手资料**（论文 / 官方实现 / 权威教材），关键结论交叉验证，
  查不到宁可不写——见 [`CLAUDE.md`](CLAUDE.md)。
- **可移植 Markdown**：在 GitHub / 任意 Markdown 阅读器 / MkDocs 三边都能读；数学用 `$...$`。

## 现状

🌱 **刚初始化**：骨架与[七部分学习路线图](docs/roadmap.md)已就位，术语表起了个头，正文章节待写。
下一步从第 1 章（MDP 骨架）动笔。

## 本地构建

在书架仓库根执行（推荐，会连带生成配色/PWA）：

```bash
python scripts/build_site.py --only rl
```

或单独调试本书：

```bash
cd books/rl && mkdocs serve
```
