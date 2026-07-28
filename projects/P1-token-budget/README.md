# 实战项目 1 · 图像 token 预算计算器

把 `token_budget.py` 里的 4 个 `TODO` 填掉，然后：

```bash
python check.py            # 自动评分（对照 transformers 真实 processor）
python check.py --verbose  # 看失败样例的详细对比
```

完整题目、验收标准、提示与常见坑：
[`docs/chapters/01-vision-basics/P1-token-budget.md`](../../docs/chapters/01-vision-basics/P1-token-budget.md)
（网页版：<https://h-jett.github.io/MultiModalLearning/chapters/01-vision-basics/P1-token-budget/>）

需要的依赖：`transformers`、`pillow`（不下载模型权重，纯 CPU 可跑）。
模型可用 `MM_VLM` 环境变量指向本地副本。
