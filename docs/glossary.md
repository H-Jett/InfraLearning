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
