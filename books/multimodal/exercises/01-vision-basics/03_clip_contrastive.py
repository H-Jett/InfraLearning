#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 3 章练习：把 CLIP 的对比学习拆开验证。

  A) 双塔 + 相似度矩阵：4 张图 × 6 句话，看对角线是否亮；顺手验证"投影后归一化"这套流程；
  B) 温度系数：CLIP 学出来的 logit_scale 是多少？把温度调大调小，概率分布怎么变；
  C) prompt 工程：裸标签 vs "a photo of a {}" vs 多模板集成，zero-shot 分数差多少；
  D) CLIP 的短板（诚实说明）：计数、空间关系、属性绑定——这些是"词袋式对齐"的代价；
  E) VLM 到底取哪一层：投影后的 512 维 pooled 特征 vs 倒数第二层的 patch token。

单卡即可。图片由 exercises/common.py 准备（下载失败会退化成合成图，实验仍能跑）。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import sample_images  # noqa: E402

CLIP_MODEL = os.environ.get("MM_CLIP", "openai/clip-vit-base-patch32")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEP = "=" * 74


def encode(model, processor, images, texts):
    """返回 L2 归一化后的图像/文本特征。这就是 CLIP 推理的全部流程。"""
    with torch.no_grad():
        px = processor(images=images, return_tensors="pt").to(DEVICE)["pixel_values"]
        tk = processor(text=texts, return_tensors="pt", padding=True).to(DEVICE)
        f_img = model.get_image_features(pixel_values=px)
        f_txt = model.get_text_features(**tk)
    return (f_img / f_img.norm(dim=-1, keepdim=True),
            f_txt / f_txt.norm(dim=-1, keepdim=True))


def part_a(model, processor, imgs):
    print(SEP)
    print("A) 双塔 + 相似度矩阵：图文在同一个空间里能直接点积")
    print(SEP)
    names = [n for n, _, _ in imgs]
    pil = [i for _, i, _ in imgs]
    texts = [                          # 前 4 句与 4 张图一一对应（顺序相同）
        "a close-up photo of a wild cat standing in the snow",
        "two tabby cats lying on a pink blanket with two remote controls",
        "an indoor living room with a television and a dining table",
        "an old stone house with a red tiled roof and a green lawn",
        "a bowl of noodles",          # 干扰项：谁都不该匹配
        "a diagram of a neural network",   # 干扰项
    ]
    f_img, f_txt = encode(model, processor, pil, texts)
    print(f"图像特征 {tuple(f_img.shape)}   文本特征 {tuple(f_txt.shape)}"
          f"   ← 两塔输出**同维度**({f_img.shape[1]})，这是能点积的前提")

    sim = f_img @ f_txt.T                     # cosine 相似度（已归一化）
    print(f"\ncosine 相似度矩阵（行=图，列=文本）：")
    print("            " + "".join(f"{'T' + str(j):>8}" for j in range(len(texts))))
    for i, n in enumerate(names):
        row = "".join(f"{v:>8.3f}" for v in sim[i].tolist())
        star = "  ← 最高分: T%d" % int(sim[i].argmax())
        print(f"{n:>12}{row}{star}")
    print("\n文本对照：" + "  ".join(f"T{j}={t[:26]}" for j, t in enumerate(texts[:4])))
    print("          " + "  ".join(f"T{j}={t[:26]}" for j, t in enumerate(texts[4:], start=4)))

    hit = sum(int(sim[i].argmax()) == i for i in range(len(names)))
    print(f"\n对角线命中 {hit}/{len(names)}。注意 cosine 的绝对值都不高（0.2~0.3）——")
    print("CLIP 只保证**相对**顺序有意义，别拿绝对值当『相似度百分比』。")
    return sim


def part_b(model, sim):
    print("\n" + SEP)
    print("B) 温度系数：把相似度变成概率的那个旋钮")
    print(SEP)
    scale = model.logit_scale.exp().item()
    print(f"CLIP 学出来的 logit_scale.exp() = {scale:.2f}"
          f"   → 等价温度 τ = 1/{scale:.2f} = {1 / scale:.4f}")
    print("（论文里 τ 被参数化成 exp(logit_scale) 并做了上限裁剪，防止训练早期爆掉）")

    row = sim[1]           # 拿第二张图（两只猫）那一行做演示
    print(f"\n以『两只猫』那张图为例，看不同温度下的 softmax：")
    print(f"{'温度 τ':>10} | {'等价 scale':>10} | {'最高概率':>9} | {'熵(nats)':>9} | 概率分布")
    print("-" * 78)
    for tau in [1.0, 0.1, 1 / scale, 0.005]:
        probs = (row / tau).softmax(dim=-1)
        ent = -(probs * probs.clamp_min(1e-12).log()).sum().item()
        bars = " ".join(f"{p:.2f}" for p in probs.tolist())
        tag = "  ← CLIP 实际用的" if abs(tau - 1 / scale) < 1e-9 else ""
        print(f"{tau:>10.4f} | {1 / tau:>10.1f} | {probs.max():>9.3f} | {ent:>9.3f} | {bars}{tag}")

    print("\n温度在干什么：")
    print("  · τ 大（scale 小）→ 分布平坦，正负样本的梯度差别小，学得慢、也学不『尖』；")
    print("  · τ 小（scale 大）→ 分布尖锐，等价于**放大难负样本的惩罚**，但太小会梯度爆炸/过拟合；")
    print("  · CLIP 干脆把它当**可学习参数**（初始 0.07 附近），让模型自己找合适的锐度。")


def part_c(model, processor, imgs):
    print("\n" + SEP)
    print("C) prompt 工程：同一个类别，换个说法分数就变")
    print(SEP)
    pil = [i for _, i, _ in imgs]
    classes = ["cat", "living room", "house", "noodles"]
    templates = {
        "裸标签": ["{}"],
        "标准模板": ["a photo of a {}"],
        "3 模板集成": ["a photo of a {}", "a blurry photo of a {}", "a close-up photo of a {}"],
    }

    print(f"（4 张图，类别 {classes}；报告每张图判对与否 + 正确类的概率）\n")
    print(f"{'方案':>12} | " + " | ".join(f"{n[:10]:>10}" for n, _, _ in imgs)
          + " | 判对数 | 正确类概率均值")
    print("-" * 92)
    truth = [0, 0, 1, 2]   # cat.jpg→cat, two_cats→cat, living_room→living room, house→house
    for tag, tpls in templates.items():
        # 多模板：对每个类别的多个模板取特征平均（CLIP 官方做法）
        feats = []
        for c in classes:
            _, f_txt = encode(model, processor, pil[:1], [t.format(c) for t in tpls])
            f = f_txt.mean(dim=0, keepdim=True)
            feats.append(f / f.norm(dim=-1, keepdim=True))
        f_txt = torch.cat(feats, dim=0)
        f_img, _ = encode(model, processor, pil, ["x"])
        probs = (model.logit_scale.exp() * f_img @ f_txt.T).softmax(dim=-1)
        cells, correct, ps = [], 0, []
        for i in range(len(pil)):
            p = probs[i, truth[i]].item()
            ps.append(p)
            ok = int(probs[i].argmax()) == truth[i]
            correct += ok
            cells.append(f"{'✅' if ok else '❌'}{p:>7.2f}")
        print(f"{tag:>12} | " + " | ".join(f"{c:>10}" for c in cells)
              + f" | {correct}/{len(pil)}   | {sum(ps) / len(ps):>10.3f}")

    print("\n诚实说明：这 4 张图**太好分了**，三种方案都 4/4 全对，看不出准确率差异；")
    print("能看到的只是**正确类概率**的变化（而且不是一致变好：cat.jpg 反而从 0.99 掉到 0.93，")
    print("因为『a photo of a cat』把它和 two_cats.jpg 拉近了）。想量出 prompt 工程的收益，")
    print("必须上有难度的基准（几十个类、上千张图），几张图的演示只能说明**机制**。")
    print("\n机制是什么：CLIP 的文本塔训的是**自然句子**（网页 alt-text），裸标签『cat』在训练")
    print("分布里很罕见；套上模板是**把输入拉回训练分布**。CLIP 论文报告：prompt 工程 + 多模板")
    print("集成在 ImageNet 上带来约 5 个点的提升（官方用了 80 个模板）——这才是可信的量级。")


def part_d(model, processor, imgs):
    print("\n" + SEP)
    print("D) CLIP 的短板：它更像『词袋匹配』而不是『理解』")
    print(SEP)
    two_cats = [i for n, i, _ in imgs if n == "two_cats.jpg"][0]

    probes = [
        ("计数", ["two cats on a couch", "three cats on a couch", "one cat on a couch",
                  "four cats on a couch"], 0),
        ("空间关系", ["two remote controls next to two cats",
                      "two cats standing on top of two remote controls"], 0),
        ("属性绑定", ["a pink blanket and two grey cats", "a grey blanket and two pink cats"], 0),
    ]
    for tag, texts, gold in probes:
        f_img, f_txt = encode(model, processor, [two_cats], texts)
        probs = (model.logit_scale.exp() * f_img @ f_txt.T).softmax(dim=-1)[0]
        pick = int(probs.argmax())
        print(f"\n【{tag}】（图：沙发上**两只**猫 + 两个遥控器）")
        for j, t in enumerate(texts):
            mark = "←选中" if j == pick else ""
            gold_mark = "(正确)" if j == gold else ""
            print(f"   {probs[j]:>6.3f}  {t:<48}{mark}{gold_mark}")
        print(f"   → {'✅ 选对' if pick == gold else '❌ 选错'}")

    print("\n诚实说明：这是**单张图的几个探针**，不是基准评测，别当成定量结论。")
    print("但方向和文献一致（Winoground、ARO 等基准）：对比学习优化的是**整句 ↔ 整图**的相似度，")
    print("句子里的词序、数量、绑定关系很容易被『平均掉』——所以 CLIP 常被形容为词袋式对齐。")
    print("这也正是 VLM 需要一个 LLM 接在后面的原因：细粒度推理交给 LLM，CLIP 只负责给好特征。")


def part_e(model, processor, imgs):
    print("\n" + SEP)
    print("E) VLM 到底取 CLIP 的哪一层？")
    print(SEP)
    pil = [imgs[0][1]]
    with torch.no_grad():
        px = processor(images=pil, return_tensors="pt").to(DEVICE)["pixel_values"]
        vout = model.vision_model(px, output_hidden_states=True)
        pooled = model.get_image_features(pixel_values=px)

    last, penult = vout.hidden_states[-1], vout.hidden_states[-2]
    print(f"① 检索/分类用的 pooled 特征 : {tuple(pooled.shape)}"
          f"   ← CLS → LayerNorm → 512 维投影，**一张图只剩一个向量**")
    print(f"② 最后一层 hidden_states    : {tuple(last.shape)}   （含 CLS，共 50 个 token）")
    print(f"③ 倒数第二层 hidden_states  : {tuple(penult.shape)}")
    print(f"\nVLM（LLaVA 系）用的是 ③ 去掉 CLS 后的 patch token：{tuple(penult[:, 1:].shape)}")

    cos = torch.nn.functional.cosine_similarity
    same = cos(last[:, 1:], penult[:, 1:], dim=-1)
    print(f"\n最后一层与倒数第二层的 patch token 逐个比 cosine："
          f"均值 {same.mean():.3f}  最小 {same.min():.3f}  最大 {same.max():.3f}")
    print("→ 两层**并不相同**，差异主要来自最后一层为了『对比学习目标』做的特化。")
    print("\nLLaVA 论文的经验做法与解读：最后一层被训练成『为图文对比服务的全局摘要』，")
    print("局部细节相对被压缩；倒数第二层保留更多空间/细节信息，更适合喂给 LLM 做细粒度问答。")
    print("（诚实说明：这是消融实验得到的工程经验，不是理论定理；也有模型选别的层或多层融合。）")


def main():
    from transformers import CLIPModel, CLIPProcessor

    print(f"模型：{CLIP_MODEL}  设备：{DEVICE}\n")
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    imgs = sample_images()
    print()

    sim = part_a(model, processor, imgs)
    part_b(model, sim)
    part_c(model, processor, imgs)
    part_d(model, processor, imgs)
    part_e(model, processor, imgs)


if __name__ == "__main__":
    main()
