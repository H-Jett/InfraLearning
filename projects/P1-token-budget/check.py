#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实战项目 1 的自动评分脚本。

对照真源 = transformers 里 Qwen2.5-VL 的**真实 image processor 输出**（不是我的实现），
所以过了就是真的对了。

用法：
    python check.py              # 打分
    python check.py --verbose    # 打印失败样例的详细对比
    python check.py --n 500      # 随机样例数（默认 200）

题目：docs/projects/P1-token-budget.md
"""

import argparse
import os
import random
import sys

VLM_MODEL = os.environ.get("MM_VLM", "Qwen/Qwen2.5-VL-3B-Instruct")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_student():
    try:
        import token_budget
    except Exception as e:
        print(f"❌ 无法 import token_budget.py：{e}")
        sys.exit(1)
    return token_budget


def make_image(h, w):
    from PIL import Image
    return Image.new("RGB", (w, h), (128, 128, 128))


class Scorer:
    def __init__(self, verbose):
        self.verbose = verbose
        self.items = []

    def add(self, name, ok, detail=""):
        self.items.append((name, ok, detail))
        print(f"  {'✅' if ok else '❌'} {name}" + (f"  {detail}" if detail else ""))

    def note(self, msg):
        if self.verbose:
            print(f"      · {msg}")

    def report(self):
        passed = sum(1 for _, ok, _ in self.items if ok)
        total = len(self.items)
        print("\n" + "=" * 60)
        print(f"得分：{passed}/{total}")
        if passed == total:
            print("🎉 全部通过！把实现和这份输出发我，我来 review。")
        else:
            print("还没全过。失败项看上面的 ❌，可以加 --verbose 看详细对比。")
        print("=" * 60)
        return 0 if passed == total else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--n", type=int, default=200, help="随机样例数")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoProcessor

    print(f"加载对照真源：{VLM_MODEL}（只读 processor/config，不下权重）")
    proc = AutoProcessor.from_pretrained(VLM_MODEL)
    ip = proc.image_processor
    cfg = AutoConfig.from_pretrained(VLM_MODEL)
    tcfg = getattr(cfg, "text_config", cfg)

    P, MERGE = ip.patch_size, ip.merge_size
    FACTOR = P * MERGE
    MINP, MAXP = ip.min_pixels, ip.max_pixels
    L = tcfg.num_hidden_layers
    KVH = tcfg.num_key_value_heads
    HD = getattr(tcfg, "head_dim", None) or tcfg.hidden_size // tcfg.num_attention_heads

    print(f"  patch_size={P} merge_size={MERGE} factor={FACTOR} "
          f"min_pixels={MINP} max_pixels={MAXP}")
    print(f"  LLM: layers={L} kv_heads={KVH} head_dim={HD}\n")

    tb = load_student()
    s = Scorer(args.verbose)
    kw = dict(factor=FACTOR, min_pixels=MINP, max_pixels=MAXP)

    def truth_grid(h, w):
        """真源：processor 实际产出的 (grid_h, grid_w)。"""
        enc = ip(images=make_image(h, w), return_tensors="pt")
        _, gh, gw = enc["image_grid_thw"][0].tolist()
        return gh, gw

    rng = random.Random(0)
    cases = [(rng.randint(20, 2200), rng.randint(20, 2200)) for _ in range(args.n)]

    # ---------------- A) smart_resize 与真实 processor 逐例对齐 ----------------
    print("A) smart_resize vs 真实 processor")
    bad = []
    for h, w in cases:
        gh, gw = truth_grid(h, w)
        want = (gh * P, gw * P)
        try:
            got = tuple(tb.smart_resize(h, w, **kw))
        except NotImplementedError:
            print("  ❌ smart_resize 还没实现（TODO(1)）")
            bad = cases
            break
        except Exception as e:
            bad.append((h, w, want, f"异常 {type(e).__name__}: {e}"))
            continue
        if got != want:
            bad.append((h, w, want, got))
    for b in bad[:8]:
        s.note(f"输入 h={b[0]} w={b[1]} → 期望 {b[2]}，你的 {b[3]}" if len(b) == 4 else str(b))
    s.add(f"A. {args.n} 组随机尺寸完全一致", len(bad) == 0,
          f"（偏差 {len(bad)} 组）" if bad else "")

    # ---------------- B) token 数 vs input_ids 里的 image_pad 个数 ----------------
    print("\nB) vision_tokens vs 实测 <|image_pad|> 个数")
    pad_id = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    sub = cases[:20]
    bad = []
    for h, w in sub:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": "hi"}]}]
        prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = proc(text=[prompt], images=[make_image(h, w)], return_tensors="pt")
        want = int((inputs["input_ids"][0] == pad_id).sum())
        try:
            got = tb.vision_tokens(h, w, merge_size=MERGE, **kw)
        except NotImplementedError:
            print("  ❌ vision_tokens 还没实现（TODO(2)）")
            bad = sub
            break
        except Exception as e:
            bad.append((h, w, want, f"异常 {type(e).__name__}: {e}"))
            continue
        if got != want:
            bad.append((h, w, want, got))
    for b in bad[:8]:
        s.note(f"输入 h={b[0]} w={b[1]} → 期望 {b[2]} tokens，你的 {b[3]}" if len(b) == 4 else str(b))
    s.add(f"B. {len(sub)} 组的 token 数完全一致", len(bad) == 0,
          f"（偏差 {len(bad)} 组）" if bad else "")

    # ---------------- C) 边界情况 ----------------
    print("\nC) 边界情况（极小 / 极长 / 超大）")
    edge = [(16, 16), (50, 4000), (4000, 50), (6000, 8000), (28, 28), (1, 1)]
    ok_all, notes = True, []
    for h, w in edge:
        try:
            rh, rw = tb.smart_resize(h, w, **kw)
        except NotImplementedError:
            ok_all = False
            notes.append("smart_resize 未实现")
            break
        except Exception as e:
            ok_all = False
            notes.append(f"{w}x{h} 抛异常 {type(e).__name__}")
            continue
        problems = []
        if rh % FACTOR or rw % FACTOR:
            problems.append(f"未对齐到 {FACTOR}")
        if rh < FACTOR or rw < FACTOR:
            problems.append(f"边长小于 {FACTOR}")
        if not (MINP <= rh * rw <= MAXP):
            problems.append(f"像素数 {rh * rw} 越界 [{MINP}, {MAXP}]")
        # 与真源对照（processor 能处理的话）
        try:
            gh, gw = truth_grid(h, w)
            if (rh, rw) != (gh * P, gw * P):
                problems.append(f"与 processor 不一致（期望 {(gh * P, gw * P)}，你的 {(rh, rw)}）")
        except Exception:
            pass  # processor 自己都拒绝的输入，不强求
        if problems:
            ok_all = False
            notes.append(f"{w}x{h}: " + "; ".join(problems))
    for n in notes[:8]:
        s.note(n)
    s.add("C. 边界情况全部合规", ok_all, f"（{len(notes)} 处问题）" if notes else "")

    # ---------------- D) KV cache 字节数 ----------------
    print("\nD) kv_cache_bytes")
    try:
        got = tb.kv_cache_bytes(1000, num_layers=L, num_kv_heads=KVH, head_dim=HD,
                                bytes_per_elem=2)
        want = 2 * L * KVH * HD * 1000 * 2
        s.add("D. 1000 token 的 KV cache 字节数正确", got == want,
              f"（期望 {want}，你的 {got}）" if got != want else f"= {want / 2**20:.2f} MiB")
    except NotImplementedError:
        s.add("D. kv_cache_bytes", False, "还没实现（TODO(3)）")
    except Exception as e:
        s.add("D. kv_cache_bytes", False, f"异常 {type(e).__name__}: {e}")

    # ---------------- E) 反向求解 ----------------
    print("\nE) max_pixels_for_budget（反向求解）")
    scenarios = [(8 * 2**30, 32, 200), (2 * 2**30, 8, 100), (24 * 2**30, 64, 500)]
    ok_all, notes = True, []
    for budget, conc, txt in scenarios:
        try:
            mp = tb.max_pixels_for_budget(budget, concurrency=conc, num_layers=L,
                                          num_kv_heads=KVH, head_dim=HD,
                                          bytes_per_elem=2, factor=FACTOR,
                                          merge_size=MERGE, text_tokens=txt)
        except NotImplementedError:
            ok_all = False
            notes.append("max_pixels_for_budget 未实现")
            break
        except Exception as e:
            ok_all = False
            notes.append(f"预算 {budget >> 30}GiB 抛异常 {type(e).__name__}")
            continue
        # 代回验证：用一张"刚好撑满 mp"的图
        n_tok = mp // (FACTOR * FACTOR)
        used = 2 * L * KVH * HD * (n_tok + txt) * 2 * conc
        if used > budget:
            ok_all = False
            notes.append(f"预算 {budget >> 30}GiB/{conc}路: 超预算（用了 {used / 2**30:.2f} GiB）")
        elif used < budget * 0.9:
            ok_all = False
            notes.append(f"预算 {budget >> 30}GiB/{conc}路: 太保守（只用了 {used / budget:.0%}）")
        else:
            s.note(f"预算 {budget >> 30}GiB/{conc}路 → max_pixels={mp}"
                   f"（{n_tok} tok/图，用掉 {used / budget:.0%}）")
        if mp % (FACTOR * FACTOR):
            ok_all = False
            notes.append(f"预算 {budget >> 30}GiB: max_pixels 不是 {FACTOR}² 的整数倍")
    for n in notes[:8]:
        s.note(n)
    s.add("E. 三个场景的反向求解都合规", ok_all, f"（{len(notes)} 处问题）" if notes else "")

    sys.exit(s.report())


if __name__ == "__main__":
    main()
