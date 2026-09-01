# 第 9 章 · 思考题参考答案

> 只记录问题与参考答案。对应正文《第 9 章 9.6 思考题》。

<a id="q1"></a>

## 1. 70B、单用户交互、长 system prompt 的客服场景，优先上哪几个？

**参考答案**（低并发 + 长共享前缀 + 大模型）：

1. **前缀复用 (Prefix Caching / RadixAttention)**：长 system prompt 在成千上万请求间完全共享，
   只算一次、后续复用，省掉海量重复 prefill，直接降 TTFT（第 5 章）。
2. **weight-only 量化 (W4A16)**：70B 的 decode 是 bandwidth-bound（TPOT 下限 ~78ms），压权重到 int4
   ≈ 按字节比线性提速 decode，还省显存（第 6 章）。
3. **投机解码 (Speculative / EAGLE)**：单用户 = 低并发，目标模型 memory-bound、算力有富余，
   "免费验证"成立，可无损降 TPOT（第 7 章）。
4. 底座当然是 **PagedAttention + continuous batching**（任何引擎自带）。

场景特征（低并发、延迟敏感、长共享前缀、大模型）正好命中前缀复用 + 量化 + 投机解码三件套。

<a id="q2"></a>

## 2. 离线批量处理（高吞吐、不在乎延迟、prompt 各异），上哪些？不上哪些？

**参考答案**：

**该上**：
- **大 batch + continuous batching**：不在乎延迟就把并发往拐点甚至更高怼，最大化吞吐（第 3 章）；
- **PagedAttention**：高并发下 KV cache 暴涨，靠分页塞下尽量多序列（第 4 章）；
- **量化（权重 / KV）**：腾显存装更多并发、也帮 compute（W8A8）（第 6 章）。

**明确不该上**：
- **投机解码**：高并发下目标已 compute-bound，"免费验证"不再免费，收益很小甚至拖累（第 7 章）；
- **前缀复用**：每条 prompt 各不相同、无公共前缀，缓存命中率≈0，用不上（第 5 章）。

要点：**离线高吞吐场景的优化组合，和在线低延迟场景几乎相反**——判断依据是"decode 处于 memory-bound
还是 compute-bound"、"有没有共享前缀"。

<a id="q3"></a>

## 3. 为什么三大引擎都实现 PagedAttention + continuous batching，却在前缀复用/编译/可编程上各有侧重？

**参考答案**：因为 **PagedAttention + continuous batching 是吞吐的"地基"**——它们直接决定 GPU 利用率
和显存能塞多少并发，是任何高性能引擎的必备门槛，所以人人都做、且做法趋同。

而上层是**差异化竞争**，对应不同目标场景：
- **vLLM**：通用易用、生态大，做"默认选择"；
- **SGLang**：押注 RadixAttention（前缀复用）+ 可编程前端，主攻多轮/agent/结构化生成；
- **TensorRT-LLM**：走 AOT 编译，在 NVIDIA 上榨极致性能，牺牲灵活性。

即：**地基相同（必备），上层按各自定位分化（差异化）**。这也说明前 8 课的技术里，有的是"人人必备"
（batching、paging），有的是"看场景取舍"（前缀复用、投机解码、编译）。
