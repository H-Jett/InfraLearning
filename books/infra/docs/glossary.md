# 术语表 (Glossary)

> 全书术语的全称与简明解释，按主题分组。各章通过脚注或链接指到这里。
> 每个词条前有一个显式 HTML 锚点（如 `<a id="hbm"></a>`），保证 GitHub、
> Markdown 阅读器、MkDocs 三边的跳转链接都稳定一致。

## 硬件与存储

<a id="hbm"></a>
### HBM（High Bandwidth Memory，高带宽内存）
一种把多层 DRAM 芯片垂直堆叠（3D 堆叠）、通过硅通孔连接、再经中介层贴在 GPU 旁边的显存。
核心优势是**极高带宽**（超宽位宽）和**每比特能耗低**，多用于数据中心 GPU。

<a id="gddr"></a>
### GDDR（Graphics Double Data Rate，图形用双倍数据率显存）
消费级显卡常用的显存（如 GDDR6/GDDR7），芯片平铺在电路板上。带宽通常低于 HBM，但成本低。

<a id="tsv"></a>
### TSV（Through-Silicon Via，硅通孔）
垂直穿透硅芯片的导电通道，是把多层 DRAM 堆叠起来做电气连接的关键工艺，HBM 靠它实现 3D 堆叠。

<a id="interposer"></a>
### Interposer（中介层）
一块位于 GPU 芯片和 HBM 之间的硅基板，提供超多、超短的走线，把 HBM 和 GPU 紧密互联。

<a id="memory-hierarchy"></a>
### Memory Hierarchy（存储层级）
从"大而慢"到"小而快"的多级存储：显存(HBM/GDDR) → L2 缓存 → 每个 SM 的 shared memory/L1 →
寄存器 → 计算单元。越靠近计算单元越快、越小。

<a id="bandwidth"></a>
### Bandwidth（带宽）
单位时间能搬运的数据量（字节/秒），如 TB/s。决定"搬数据"的速度，是访存受限场景的瓶颈。

## 计算特性

<a id="memory-bound"></a>
### Memory-bound（访存受限）
运算的瓶颈在"搬数据"而非"算"，即等显存带宽的时间 > 计算时间。LLM 逐 token 解码就是典型。

<a id="compute-bound"></a>
### Compute-bound（计算受限）
运算的瓶颈在算力，计算时间 > 搬数据时间。LLM 的 prefill 阶段（一次处理很多 token）就是典型。

<a id="arithmetic-intensity"></a>
### Arithmetic Intensity（算术强度）
一次运算的浮点运算数 ÷ 需要搬运的字节数（FLOPs/Byte）。低于 GPU 的"临界强度"就是访存受限，
高于则是计算受限。是判断 memory-bound / compute-bound 的定量工具。

<a id="launch-bound"></a>
### Launch-bound / CPU-bound（发射受限）
计算量太小时，耗时被"CPU 发射 GPU 指令"的固定开销主导，GPU 算得比 CPU 喂得还快而空等。

## CUDA 与并发

<a id="cuda"></a>
### CUDA（Compute Unified Device Architecture）
NVIDIA 的 GPU 并行计算平台与编程模型，让程序能把计算任务下发到 GPU 上执行。

<a id="kernel"></a>
### Kernel（核函数）
在 GPU 上并行执行的一个函数/计算任务。一次模型前向会发射几十上百个 kernel（每层若干个）。

<a id="async"></a>
### Asynchronous Execution（异步执行）
CPU 把 kernel 发射给 GPU 后**不等它算完**就继续往下走。所以直接计时测到的可能只是"发射"耗时。

<a id="synchronize"></a>
### torch.cuda.synchronize()（同步 / 屏障）
一个**屏障 (barrier)**：CPU 走到这里必须停下，等 GPU 把已排队的任务全部算完才继续。
它**不是**"启动 GPU"的信号；若队列本就为空，会立即返回。

<a id="stream"></a>
### CUDA Stream（流）
GPU 上的一条任务队列（"车道"）。同一个流内任务按序执行，不同流之间可并发，
用于让计算与通信等操作重叠 (overlap)。`stream.synchronize()` 只等这一条流。

<a id="nccl"></a>
### NCCL（NVIDIA Collective Communications Library）
NVIDIA 的多卡集合通信库，提供 AllReduce、AllGather 等原语，是分布式训练卡间通信的底座。

## 推理

<a id="autoregressive"></a>
### Autoregressive（自回归）
逐 token 生成：每次预测下一个 token，再把它接到序列末尾继续预测，每个 token 依赖之前所有 token。

<a id="prefill"></a>
### Prefill（预填充）
推理第一阶段：把整段 prompt 一次性、并行地喂进模型做一次前向，产出第一个新 token。计算受限。

<a id="decode"></a>
### Decode（解码）
推理第二阶段：从第一个新 token 起逐个生成，每步只处理一个 token（借助 KV Cache）。访存受限。

<a id="ttft"></a>
### TTFT（Time To First Token，首 token 延迟）
从收到请求到吐出第一个 token 的时间，主要由 prefill 决定。

<a id="tpot"></a>
### TPOT（Time Per Output Token，每 token 输出耗时）
解码阶段每个后续 token 的间隔，也叫 ITL (Inter-Token Latency)。理论下限 ≈ 模型大小 ÷ 显存带宽。

<a id="kv-cache"></a>
### KV Cache（键值缓存）
缓存历史 token 在注意力中算出的 Key/Value，使解码每步只需计算新 token，无需重算整段历史。
是解码提速的关键，也是长上下文/高并发时的主要显存消耗者。

<a id="mha"></a>
### MHA（Multi-Head Attention，多头注意力）
标准注意力：每个 Query 头都有自己独立的 Key/Value 头。KV cache 最大。

<a id="gqa"></a>
### GQA（Grouped-Query Attention，分组查询注意力）
多个 Query 头**共享**一组 Key/Value 头（KV 头数 < Q 头数）。在几乎不损质量的前提下把
KV cache 按比例砍小，是现代大模型的主流选择。

<a id="mqa"></a>
### MQA（Multi-Query Attention，多查询注意力）
GQA 的极端情形：**所有** Query 头共享同一组 Key/Value（KV 头数 = 1）。KV cache 最小，
但对质量影响比 GQA 大。

<a id="warmup"></a>
### Warmup（预热）
正式计时前先空跑几次，让 GPU 完成显存分配、kernel 编译/缓存等一次性冷启动开销，使测量稳定。

<a id="static-batching"></a>
### Static Batching（静态批处理）
一批请求一起进、一起出，必须等整批（最长那条）完成才能开始下一批。实现简单，但有长尾浪费
和排队延迟两个问题。

<a id="continuous-batching"></a>
### Continuous Batching（连续批处理）
又称 in-flight batching / iteration-level scheduling。把调度粒度降到每个 decode 步：完成的序列
立即退出并释放 KV cache，等待的新请求立即补进空位。GPU 始终满载，是现代推理引擎（vLLM/SGLang/TGI）
高吞吐的核心。

<a id="paged-attention"></a>
### PagedAttention
vLLM 提出的 KV cache 管理技术：借用操作系统内存分页的思路，把 KV cache 切成固定大小的 block，
按需分配、物理上可不连续，消除碎片，让有限显存装下更多并发序列。

<a id="kv-block"></a>
### Block（KV 块）
PagedAttention 里 KV cache 分页的固定大小单位，存若干个 token 的 K/V（vLLM 默认 16 个）。

<a id="block-table"></a>
### Block Table（块表）
记录一条序列的"逻辑块 → 物理块"映射的表，相当于操作系统的页表，让分散的物理块拼成连续的逻辑 KV。

<a id="fragmentation"></a>
### Fragmentation（碎片）
显存的两类浪费：**内部碎片**（分配了但没用满，如预留最大长度）、**外部碎片**（释放后留下放不下
新请求的小空洞）。PagedAttention 用定长块 + 按需分配基本消除两者。

<a id="prefix-caching"></a>
### Prefix Caching（前缀复用 / 前缀缓存）
缓存共享前缀（如同一 system prompt、对话历史）的 KV，后续以相同前缀开头的请求直接复用这份 KV、
跳过对该前缀的 prefill，从而降低 TTFT 和计算量。vLLM 的 Automatic Prefix Caching (APC) 是其实现之一。

<a id="radix-attention"></a>
### RadixAttention
SGLang 提出：用**前缀树 (radix tree)** 管理所有缓存的前缀 KV，自动匹配并复用请求间的最长公共前缀，
显存不足时按 LRU 淘汰。把前缀复用泛化到任意请求间的任意公共前缀。

## 量化

<a id="quantization"></a>
### Quantization（量化）
用更少的比特表示数值（权重/激活/KV），如 bf16→int8/int4/fp8。通过 `q=round(x/scale)+zero_point`
线性映射到整数。收益：省显存；decode 带宽受限时提速；低精度 Tensor Core 加速计算。代价是量化误差。

<a id="ptq-qat"></a>
### PTQ / QAT
PTQ (Post-Training Quantization，训练后量化)：训练完再用少量校准数据量化，简单常用（GPTQ/AWQ）。
QAT (Quantization-Aware Training，量化感知训练)：训练时就模拟量化，精度更好但成本高。

<a id="gptq"></a>
### GPTQ
一种 weight-only 训练后量化方法，逐层贪心地量化权重并最小化输出误差，int4 常近乎无损。

<a id="awq"></a>
### AWQ（Activation-aware Weight Quantization）
weight-only 量化：根据激活的重要性保护少数关键权重通道，其余低精度，int4 下精度好、速度快。

<a id="fp8"></a>
### FP8（8 位浮点，E4M3 / E5M2）
1 字节浮点格式，Hopper 起硬件原生支持。E4M3 精度高、E5M2 动态范围大。推理和训练（配合缩放）都在用。

<a id="weight-only"></a>
### Weight-only 量化（如 W4A16）/ W8A8
Weight-only：只量化权重（如 int4），计算时反量化回 fp16——主要加速大模型 decode（访存受限）。
W8A8：权重和激活都量化到 int8，用 int8 Tensor Core，连 compute-bound 的 prefill 也能提速（如 SmoothQuant）。
记号：W=weights、A=activations，数字为比特数（W4A16 = 权重 4 bit、激活 16 bit）。

## 投机解码

<a id="speculative-decoding"></a>
### Speculative Decoding（投机解码）
用一个又小又快的草稿模型先猜 K 个 token，目标（大）模型一次并行验证，接受与目标一致的最长前缀。
一次目标前向吐出多个 token，加速 decode；靠接受/拒绝采样保证输出分布严格等于目标模型——**无损**。

<a id="draft-model"></a>
### Draft Model（草稿模型）
投机解码里那个又小又快、负责提出候选 token 的模型。也可用模型自身的额外头（Medusa/EAGLE）代替。

<a id="acceptance-rate"></a>
### Acceptance Rate（接受率 α）
草稿 token 被目标模型接受的比例。α 越高（草稿越像目标），每次验证吐出的 token 越多、加速越大。

<a id="medusa"></a>
### Medusa / EAGLE
自我投机方案：不另养草稿模型，而给模型加额外预测头（Medusa）或特征层轻量自回归头（EAGLE）来产草稿。

## 服务指标

<a id="throughput"></a>
### Throughput（吞吐）
系统单位时间产出的 token 数（tok/s）或完成的请求数（QPS）。主要由 batching 决定，是系统级视角。

<a id="goodput"></a>
### Goodput（有效吞吐）
只计**满足 SLO**（延迟目标）的那部分吞吐。越过延迟拐点后，原始吞吐可能还涨，但请求超时不算数，
goodput 反而下降。调服务的真正目标是"满足 SLO 前提下最大化 goodput"，而非最大化 throughput。

<a id="slo"></a>
### SLO（Service Level Objective，服务级目标）
对服务定的量化延迟目标，通常按分位数写，如 "P99 TTFT ≤ 500ms、P99 TPOT ≤ 50ms"。

<a id="tail-latency"></a>
### Tail Latency / 分位数（P50 / P90 / P99）
延迟的分布刻画：P99 = 99% 的请求比它快，反映"最差那批用户"的体验。评估服务看 P99，不看平均——
平均会掩盖长尾，而一个慢请求能拖垮串联的调用链。

## 推理引擎

<a id="vllm"></a>
### vLLM
PagedAttention 的发源地。Python 为主、易扩展、生态最大、兼容多硬件，功能全的"默认选择"引擎。

<a id="sglang"></a>
### SGLang
招牌是 RadixAttention（前缀树自动复用）+ 可编程前端（结构化/分支/并行生成），擅长多轮对话、agent、
高前缀复用场景。

<a id="tensorrt-llm"></a>
### TensorRT-LLM
NVIDIA 官方引擎，走提前编译 (AOT) 路线：把模型编译成高度优化的引擎，配 in-flight batching，
在 NVIDIA 卡上追求极致性能，但编译流程重、灵活性低、只服务 NVIDIA。

## GPU 架构与算子

<a id="sm"></a>
### SM（Streaming Multiprocessor，流多处理器）
GPU 的基本计算单元，内含 CUDA cores、Tensor cores、warp 调度器、shared memory/L1、寄存器。
一个 block 被整体分派到一个 SM 上执行（RTX 5090 有 170 个 SM）。

<a id="warp"></a>
### Warp（线程束）
32 个线程一组，是 GPU 真正的调度/执行单位。同一 warp 内线程按 SIMT 锁步执行同一指令。

<a id="simt"></a>
### SIMT（Single Instruction, Multiple Threads）
GPU 执行模型：一个 warp 的 32 个线程同一时刻执行同一条指令、各作用于自己的数据。

<a id="warp-divergence"></a>
### Warp Divergence（束内分支发散）
同一 warp 内线程走了不同分支（if/else），硬件只能分支串行执行、另一半线程闲置，吞吐下降。
可用掩码算术（`v += c*a + (1-c)*b`）消除。

<a id="tensor-core"></a>
### Tensor Core
矩阵乘专用硬件单元，一条指令做一小块矩阵乘加，吞吐远高于 CUDA core，但用低精度（bf16/fp16/tf32/int8/fp8）。
实测 bf16 矩阵乘 ≈ fp32 CUDA core 的 3.5×。

<a id="tf32"></a>
### TF32（TensorFloat-32）
Ampere 起 Tensor Core 的一种精度：19 位有效，介于 fp32 与 bf16 之间，让 fp32 输入也能走 Tensor Core 提速。

<a id="roofline"></a>
### Roofline 模型
横轴算术强度 (FLOP/byte)、纵轴可达算力，两道屋顶 = min(峰值算力, 带宽×算术强度)。拐点算术强度 =
峰值算力/带宽（5090 实测 ≈ 151）。强度低于拐点 → memory-bound，高于 → compute-bound。

<a id="brent"></a>
### Brent 定理（work-span）
用 p 个处理器，运行时间 max(W/p, D) ≤ T_p ≤ W/p + D。W=总运算量、D=最长依赖链（深度）。
推论：再多核也快不过 D，所以并行算法要压低 depth（如树状归约把 depth 从 N 降到 log N）。

<a id="coalescing"></a>
### Memory Coalescing（访存合并）
同一 warp 的 32 个线程若访问连续内存地址，硬件把它们合并成一次（或很少几次）大内存事务，带宽拉满；
访问分散则退化成多次小事务，带宽暴跌。写 memory-bound kernel 的头号规则：让相邻线程访问相邻内存。

<a id="shared-memory"></a>
### Shared Memory（共享内存）
每个 block 独享的一块片上高速内存（比 HBM 快几十倍），block 内线程共享。经典用法是 tiling：
把数据块载入其中反复复用，减少回 HBM 的访问。

<a id="tiling"></a>
### Tiling（分块）
把一小块数据从 HBM 搬进 shared memory 一次，让线程在片上反复复用，从而减少 HBM 访问、
提高算术强度——把算子从 memory-bound 往 compute-bound 推。矩阵乘优化的核心手法。

<a id="occupancy"></a>
### Occupancy（占用率）
一个 SM 上**活跃 warp 数 / 该 SM 最大可容纳 warp 数**。占用率越高，越有足够多的 warp 供硬件切换、
掩盖访存延迟；但受每个 block 的寄存器数、shared memory 用量限制。不是越高越好，够用即可。

<a id="profiler"></a>
### Profiler（性能分析工具）
测量程序性能的工具。GPU 三层：**torch.profiler**（框架/kernel 级时间，最易用）、
**Nsight Systems / nsys**（系统时间线与 trace）、**Nsight Compute / ncu**（单 kernel 硬件计数器，
需性能计数器权限，容器里常报 `ERR_NVGPUCTRPERM`）。
