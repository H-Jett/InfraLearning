# 实战项目 1 · 图像 token 预算计算器

> **前置**：第 1 章[图像如何变成 token](../chapters/01-vision-basics/01-image-to-tokens.md)
> **难度**：★★☆☆☆ · **预计耗时**：1.5~3 小时 · **需要 GPU**：不需要

## 为什么做这个项目

第 1 章你已经**看**我算了一遍：一张图会变成多少 token、要占多少 KV cache。
但"看懂"和"会算"之间有一条沟——这个项目就是让你自己把那条公式从**理解**变成**能跑的代码**。

做完你会得到一个真正有用的小工具：**给它一批图和一个显存预算，它告诉你该把 `max_pixels` 设成多少。**
这是 VLM 上线前必做的容量规划，也是你今后调 VLM 服务时最常回答的问题。

## 能力目标

1. 能**独立复现** Qwen-VL 系列的 `smart_resize` 预处理逻辑（这是全书后面动态分辨率章节的地基）；
2. 能把"分辨率 → 视觉 token → KV cache 显存"这条链路算准到**逐字节对齐**；
3. 会**反向求解**工程问题：给定显存/延迟预算，倒推该设多大的分辨率上限。

## 你要交付什么

骨架在 `projects/P1-token-budget/token_budget.py`，把里面 4 个 `TODO` 填掉：

| # | 函数 | 要做的事 | 难度 |
|---|------|----------|------|
| 1 | `smart_resize(h, w, factor, min_pixels, max_pixels)` | 保持长宽比，把宽高对齐到 `factor` 的倍数，并把总像素压进 `[min_pixels, max_pixels]` | ★★★ |
| 2 | `vision_tokens(h, w, ...)` | 给定原图宽高，算出送进 LLM 的视觉 token 数 | ★ |
| 3 | `kv_cache_bytes(n_tokens, cfg)` | 从模型 config 算这些 token 的 KV cache 字节数 | ★★ |
| 4 | `max_pixels_for_budget(...)` | **反向**：给定显存预算和并发数，求最大允许 `max_pixels` | ★★ |

## 验收标准（硬指标）

| 项 | 标准 |
|----|------|
| A. `smart_resize` 正确性 | 在 200 组随机尺寸上，与 transformers 的 `Qwen2VLImageProcessor` 实际输出的 `image_grid_thw` **完全一致**（0 组偏差） |
| B. token 数正确性 | 同上 200 组，`vision_tokens()` 与实测 `<\|image_pad\|>` 个数**完全一致** |
| C. 边界情况 | 极小图（16×16）、极长图（4000×50）、超大图（8000×6000）都不崩、且满足 `min_pixels ≤ 像素数 ≤ max_pixels` |
| D. KV cache 计算 | 与第 1 章公式对齐，误差 0 字节 |
| E. 反向求解 | 求出的 `max_pixels` 代回去算，显存占用 ≤ 预算，且 **≥ 预算的 90%**（不能过于保守） |

## 怎么验收

```bash
cd projects/P1-token-budget
python check.py                 # 逐项打印 ✅/❌ 和总分；全过退出码 0
python check.py --verbose       # 打印每个失败样例的详细对比（调试用）
```

评分脚本用**transformers 的真实 processor** 作对照，不是对照我的实现——所以过了就是真的对了。

## 提示（卡住了再看）

**关于 Step 1（最难的一步）**：`smart_resize` 要同时满足三个约束，顺序很关键：

1. 先把 h、w 各自**四舍五入到 `factor` 的倍数**（`factor = patch_size × merge_size = 28`）；
2. 如果此时总像素**超过** `max_pixels`：按 `sqrt(h*w / max_pixels)` 等比缩小，
   然后**向下取整**到 factor 倍数（向下才能保证不超）；
3. 如果总像素**小于** `min_pixels`：按 `sqrt(min_pixels / (h*w))` 等比放大，
   然后**向上取整**到 factor 倍数；
4. 任何时候 h、w 都不能小于 `factor`。

> 想清楚一个问题：为什么第 2 步要向下取整、第 3 步要向上取整？如果都用四舍五入会怎样？
> （这就是第 1 章里"640×480 变成了 644×476、宽反而涨了 4 像素"的由来。）

**关于 Step 4**：这是一道二分或解析求解题。KV cache 与 token 数成正比、token 数与像素数成正比，
所以整条链是线性的——你可以直接解，也可以偷懒用二分（`check.py` 不管你怎么实现，只看结果）。
注意 `max_pixels` 必须是 `factor²` 的整数倍才有意义。

**别忘了**：`min_pixels`/`max_pixels` 约束的是**像素数**，不是边长；token 数是 `(h/28) × (w/28)`。

## 加分项（做完主线再挑战）

1. **命令行工具**：`python token_budget.py --dir ./some_images --model Qwen/Qwen2.5-VL-3B-Instruct --gpu-mem 8GiB --concurrency 32`
   → 打印这批图的 token 分布（min/中位数/P95/max）+ 推荐的 `max_pixels`；
2. **画图**：把这批图的 token 数分布画成直方图，标出你推荐的上限落在哪个分位；
3. **多模型对比**：同一批图，在 Qwen2.5-VL-3B / 7B 的 config 下 KV cache 差多少倍？为什么？
4. **视频**：把 `temporal_patch_size = 2` 也考虑进来，算一段 N 帧视频的 token 数。

## 常见坑

- **把 factor 当成 14**：`patch_size=14` 但送进 LLM 前有 2×2 merge，所以对齐单位是 **28**；
  token 数也要除以 `merge_size²`。
- **用 `//` 硬算 token 数**：`(h // 28) * (w // 28)` 在没对齐的尺寸上会差几个 token，
  必须先 `smart_resize` 再算。
- **KV cache 忘了乘层数、或用了 attention 头数**：要用 `num_key_value_heads`（GQA 下比 Q 头少）。
  这两个错都会让结果差几十倍。
- **只测正常尺寸**：极端长宽比（4000×50）是最容易暴露实现 bug 的地方，`check.py` 会专门测。

## 做完之后

把你的实现和 `check.py` 的输出发我，我会 review 代码风格与边界处理，并告诉你
真实工程里这个工具还差哪几块（比如多图请求、视频、prefix caching 命中率对预算的影响）。
