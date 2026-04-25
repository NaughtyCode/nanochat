# nanochat 项目深度分析文档

## 一、项目概述

**nanochat** 是由 Andrej Karpathy 开发的"最小化全栈 ChatGPT 克隆"项目，是一个在单节点 GPU 上训练大语言模型（LLM）的实验性框架。项目涵盖了从分词器训练、预训练、监督微调（SFT）、强化学习（RL）到评估、推理和聊天 UI 的完整流水线。

### 核心理念

- **单一复杂度旋钮**：只需调整 `--depth`（Transformer 层数），系统自动推导出所有其他超参数（宽度、头数、学习率、训练周期、权重衰减），实现"计算最优"（compute-optimal）训练。
- **极低成本训练 GPT-2 级别模型**：2019 年训练 GPT-2（124M 参数）约需 $43,000，nanochat 在 8×H100 上约 2 小时即可达到同等能力，成本约 $48（降低约 600 倍）。
- **全流程一体化**：`runs/speedrun.sh` 一个脚本即可完成从数据下载到最终评估的全部流程。
- **极简但现代**：代码约 4000 行 Python，但包含了 RoPE、RMSNorm、ReLU² 激活、GQA、滑动窗口注意力、Flash Attention 3、FP8 训练、Muon+AdamW 组合优化器等现代技术。

### 技术亮点一览

| 特性 | 说明 |
|------|------|
| RoPE 旋转位置编码 | 无学习参数的位置编码，支持长上下文 |
| RMSNorm | 无偏置无缩放参数的归一化，比 LayerNorm 更快 |
| ReLU² 激活函数 | MLP 中使用 ReLU 后再平方，获得更强的非线性 |
| 分组查询注意力 (GQA) | 减少 KV 缓存内存占用，提升推理效率 |
| 滑动窗口注意力 | 部分层使用短窗口注意力，节省训练计算量 |
| QK 归一化 | 对 Query 和 Key 进行归一化，稳定训练 |
| 值嵌入 (Value Embeddings) | ResFormer 风格的交替层值嵌入 |
| 残差/x0 lambda | 逐层学习残差流缩放和初始嵌入混合 |
| Smear/Backout 机制 | 相邻 token 信息混合和中间层特征消除 |
| Logit 软封顶 | 将 logits 平滑限制在 [-15, 15] 范围 |
| Flash Attention 3 | Hopper GPU 上使用 FA3，其他平台自动回退 SDPA |
| FP8 训练 | 自定义 ~150 行 FP8 Linear 层，替代 torchao |
| MuonAdamW 优化器 | 矩阵参数用 Muon（动量 + 正交化），其他用 AdamW |
| Polar Express 正交化 | 牛顿-舒尔茨迭代的改进版正交化方法 |
| NorMuon 方差缩减 | 逐神经元自适应学习率 |
| BOS 对齐最佳适配打包 | 100% 利用率的数据打包，~35% token 裁剪率 |
| BPB 评估指标 | 与词汇量无关的 bits-per-byte 评估指标 |
| 分布式优化器 (ZeRO-2 风格) | 分片优化器状态，异步通信-计算重叠 |
| 工具使用 (计算器) | 支持 Python REPL 工具调用的对话模型 |

---

## 二、核心原理

### 2.1 GPT 模型架构

nanochat 的 GPT 模型是一个自回归 Transformer 解码器，核心公式：

$$P(x_t | x_{<t}) = \text{softmax}(\text{LMHead}(\text{Transformer}(x_{<t})))$$

#### 2.1.1 Token 嵌入与位置编码

```text
输入 tokens → wte (词嵌入) → RMSNorm → Smear (相邻token混合) → Transformer Block × N → Backout → RMSNorm → LM Head → Logits
```

- **词嵌入 (wte)**：将 token ID 映射为稠密向量，使用正态分布初始化（std=0.8）。
- **位置编码**：使用 Rotary Position Embedding (RoPE)，直接作用于 Query 和 Key 向量，无需额外参数。
- **Smear 机制**：将前一个 token 的嵌入混合到当前 token，为模型提供廉价的 bigram 级别信息：
  ```
  x[t] = x[t] + λ · σ(gate(x[t][:24])) · x[t-1]
  ```
  其中 λ 是可学习的缩放因子，gate 是一个小型线性层（24→1）。

#### 2.1.2 Transformer Block

每个 Block 包含两个子层，均使用 Pre-Norm 结构：

```
x = x + Attention(RMSNorm(x))     # 注意力子层
x = resid_lambda * x + x0_lambda * x0  # 残差缩放 + 初始嵌入混合
x = x + MLP(RMSNorm(x))           # MLP 子层
```

- **残差缩放 (resid_lambda)**：每层学习一个标量，缩放残差流的幅度。初始化从 1.15 线性衰减到 1.05。
- **x0 混合 (x0_lambda)**：每层学习将初始嵌入混合回残差流的比例。初始化从 0.20 线性衰减到 0.05。
- **Backout 机制**：在最终输出前，减去中间层（n_layer//2）的残差流，消除低级特征：
  ```
  x_final = x - backout_lambda * x_mid
  ```

#### 2.1.3 因果自注意力 (CausalSelfAttention)

```python
Q, K, V = Linear(x)  # 投影
Q, K = RoPE(Q), RoPE(K)  # 旋转位置编码
Q, K = RMSNorm(Q), RMSNorm(K)  # QK 归一化
Q, K = 1.2 * Q, 1.2 * K  # 锐化注意力分布
# GQA: Q 头数 ≥ K/V 头数，训练时可选模式
# 值嵌入 (VE): V = V + gate · value_embedding(token_ids)
Output = FlashAttention(Q, K, V, causal=True, window=window_size)
Output = Linear(Output)  # 输出投影
```

关键设计决策：
- **FA3 原生布局**：使用 (B, T, H, D) 布局，无需转置即可直接输入 FA3。
- **滑动窗口模式**：通过 `window_pattern` 字符串（如 "SSSL"）控制每层的注意力窗口，最后一层始终为全上下文。
- **QK 缩放因子 1.2**：在 Q 和 K 之间分散缩放因子，使注意力分布更锐利。

#### 2.1.4 MLP

```python
x = Linear(x)       # 扩展 4×
x = ReLU(x)²        # ReLU 后再平方
x = Linear(x)       # 投射回原始维度
```

使用 ReLU²（即 ReLU 后平方）替代 GELU，这是基于 [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) 等研究的改进。

#### 2.1.5 Logit 软封顶

```python
logits = softcap * tanh(logits / softcap)  # softcap = 15
```

将 logits 平滑压缩到 [-15, 15] 范围，防止极端值，稳定训练。

### 2.2 GPTConfig 配置系统

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048      # 最大序列长度
    vocab_size: int = 32768       # 词汇量
    n_layer: int = 12             # Transformer 层数
    n_head: int = 6               # Query 头数
    n_kv_head: int = 6            # Key/Value 头数 (GQA)
    n_embd: int = 768             # 嵌入维度
    window_pattern: str = "SSSL"  # 滑动窗口模式
```

### 2.3 分词器 (Tokenizer)

使用 GPT-4 风格的 BPE（字节对编码）分词器，两种实现：

1. **HuggingFace Tokenizer**（训练+推理，基于 `tokenizers` 库）
2. **RustBPETokenizer**（基于 `rustbpe` 训练 + `tiktoken` 高效推理，默认使用）

#### 特殊 Token

| Token | 用途 |
|-------|------|
| `<\|bos\|>` | 文档开始标记 |
| `<\|user_start\|>` / `<\|user_end\|>` | 用户消息边界 |
| `<\|assistant_start\|>` / `<\|assistant_end\|>` | 助手消息边界 |
| `<\|python_start\|>` / `<\|python_end\|>` | Python 工具调用边界 |
| `<\|output_start\|>` / `<\|output_end\|>` | 工具输出边界 |

#### 对话渲染

`render_conversation()` 将对话转换为 token 序列和 loss mask：
- **mask=1**：助手生成的文本 token（训练时计算损失）
- **mask=0**：用户消息、特殊 token、工具输出（不计算损失）

#### 分割模式

```regex
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
```

与 GPT-4 的区别：数字匹配合并为 `{1,2}` 而非 `{1,3}`，对于 32K 词汇量更优。

### 2.4 训练原理

#### 2.4.1 缩放定律 (Scaling Laws)

nanochat 使用实验得出的缩放定律自动确定最优训练配置：

1. **最优训练 Token 数**：`target_tokens = target_param_data_ratio × scaling_params`
   - `target_param_data_ratio` 默认为 12（Chinchilla 为 20）
   - `scaling_params` = transformer_matrices + lm_head（实验证明这两个的和产生最清晰的缩放曲线）

2. **最优批大小**：遵循 Power Lines 论文，`B_opt ∝ D^0.383`
   ```python
   predicted_batch_size = B_REF * (target_tokens / D_REF) ** 0.383
   ```

3. **学习率缩放**：`η ∝ √(B/B_ref)` （AdamW 标准缩放）

4. **权重衰减缩放**：基于 T_epoch 框架
   ```python
   λ = λ_ref · √(B/B_ref) · (D_ref/D)
   ```

#### 2.4.2 muP 风格迁移

参考模型 d12 的所有超参数经过精心调优，然后通过缩放定律自动迁移到更深的模型（d20, d24 等）。

#### 2.4.3 优化器：MuonAdamW

**Muon (MomentUm Orthogonalized by Newton-schulz)** 用于 2D 矩阵参数：

```
1. Nesterov 动量：g = β·buffer + (1-β)·grad; buffer = g
2. 归一化：X = g / (||g||·1.01 + ε)
3. Polar Express 正交化（替代 Newton-Schulz）：
   for a, b, c in coefficients:
       A = X @ X^T  (或 X^T @ X 对于 tall 矩阵)
       B = b*A + c*(A@A)
       X = a*X + B@X (或 X@B)
4. NorMuon 方差缩减：
   v_mean = mean(g²)
   buffer.lerp_(v_mean, 1-β₂)
   step_size = 1/sqrt(max(buffer, ε))
   g *= step_size * (||v||/||scaled_v||)
5. 谨慎更新：
   mask = (g * param) >= 0
   param -= lr * g + lr * wd * param * mask
```

**AdamW** 用于嵌入、lm_head 和标量参数：

```
1. 权重衰减：p *= (1 - lr*wd)
2. 动量更新：exp_avg.lerp_(grad, 1-β₁)
3. 二阶矩：exp_avg_sq.lerp_(grad², 1-β₂)
4. 偏差校正：bias = 1 - β^step
5. 更新：p += -lr/bias · exp_avg / (√(exp_avg_sq/bias) + ε)
```

**Polar Express** 相比 Newton-Schulz 的优势：
- 更好的收敛性：Newton-Schulz 在区间边缘不收敛，Polar Express 通过优化系数解决了这个问题
- 五次迭代系数经过精心计算，最大化零点斜率

**分布式版本 (DistMuonAdamW)** 使用 3 阶段异步通信：

```
阶段 1: 启动所有异步 reduce 操作 (reduce_scatter / all_reduce)
阶段 2: 等待 reduce 完成，计算更新，启动 gather
阶段 3: 等待 gather 完成，拷贝回原始参数
```

AdamW 采用 ZeRO-2 风格分片：
- 小参数 (<1024 元素)：all_reduce 梯度，每个 rank 存完整状态
- 大参数：reduce_scatter 梯度，每个 rank 只更新自己的分片，然后 all_gather

Muon 分组分片：
- 同形状参数堆叠在一起，按 rank 分片
- 每个 rank 只对自己拥有的参数执行 Muon 更新

#### 2.4.4 FP8 训练

自定义的实现（~150 行），直接使用 `torch._scaled_mm`：

```
前向：输入(e4m3) @ 权重(e4m3) → 输出
反向1：梯度输出(e5m2) @ 权重(e4m3) → 梯度输入
反向2：梯度输出.T(e5m2) @ 输入(e4m3) → 梯度权重
```

- **e4m3**（4 位指数，3 位尾数）：用于输入和权重（精度更高）
- **e5m2**（5 位指数，2 位尾数）：用于梯度（范围更广）
- **张量级缩放**：一个张量一个缩放因子（而非逐行）
- 与 torchao 的区别：使用 `autograd.Function` 包装，`torch.compile` 视为不透明节点

#### 2.4.5 学习率调度

```
warmup:     线性从 0 增长到 1.0 (默认 40 步)
constant:   保持 1.0
warmdown:   线性衰减到 final_lr_frac (默认 0.05)
```

Momentum 调度（Muon）：
```
warmup:     从 0.85 增长到 0.97 (400 步)
constant:   保持 0.97
warmdown:   从 0.97 衰减到 0.90
```

权重衰减调度：
```
余弦衰减到零：wd(t) = wd_0 · 0.5 · (1 + cos(π·t/T))
```

### 2.5 数据加载与打包

nanochat 使用 **BOS 对齐最佳适配（Best-Fit）打包** 算法：

```
对于每个序列行：
  1. 从缓冲区选最大的能完整放入的文档
  2. 重复直到没有文档能完整放入
  3. 裁剪最短的文档填满剩余空间
结果：100% 利用率（无填充），约 35% token 被裁剪
```

特点：
- 每行以 BOS token 开头
- 每个 token 可以一直 attend 回 BOS
- 支持 DDP 分片（每个 rank 处理不同的 row group）
- 支持断点续训（通过 `pq_idx`, `rg_idx`, `epoch` 跟踪位置）

### 2.6 推理引擎

`Engine` 类实现了高效推理：

1. **批大小为 1 的预填充**：先对 prompt token 进行一次前向传播
2. **KV 缓存复制**：将预填充的 KV 缓存复制到 `num_samples` 个样本
3. **逐 token 生成**：每步为每个样本采样一个 token
4. **工具使用状态机**：
   - 检测 `<|python_start|>` 进入 Python 模式
   - 累积 Python 表达式 token
   - 检测 `<|python_end|>` 执行计算器
   - 将结果作为 `<|output_start|>` + result + `<|output_end|>` 强制注入

### 2.7 评估体系

#### CORE 指标（基座模型）
基于 DCLM 论文的评估框架，支持三种任务类型：
- **多选题**：计算每个选项的平均损失，选最小损失的选项
- **Schema**：公共后缀的不同前缀
- **语言建模**：精确匹配 continuation

CORE = mean(centered_results)，其中 centered_result = (accuracy - 0.01×random) / (1 - 0.01×random)

#### BPB（Bits Per Byte）
与词汇量无关的损失指标：
```
BPB = total_nats / (ln(2) × total_bytes)
```
需要 `token_bytes.pt`（每个 token ID 的字节数）来归一化。

#### ChatCORE 指标（对话模型）
6 个任务的均值中心准确率：ARC-Easy, ARC-Challenge, MMLU, GSM8K, HumanEval, SpellingBee
- 分类任务（ARC, MMLU）：比较各选项字母的 logits
- 生成式任务（GSM8K, HumanEval, SpellingBee）：采样后评估

---

## 三、架构设计

### 3.1 项目目录结构

```
nanochat/
├── nanochat/              # 核心库（14 个 Python 文件）
│   ├── gpt.py             # GPT 模型定义
│   ├── tokenizer.py       # BPE 分词器
│   ├── dataset.py         # 预训练数据集工具
│   ├── dataloader.py      # 分布式数据加载器
│   ├── engine.py          # 推理引擎
│   ├── optim.py           # MuonAdamW 优化器
│   ├── flash_attention.py # Flash Attention 统一接口
│   ├── fp8.py             # FP8 训练
│   ├── checkpoint_manager.py # 检查点管理
│   ├── common.py          # 公共工具
│   ├── core_eval.py       # CORE 指标评估
│   ├── loss_eval.py       # BPB 评估
│   ├── execution.py       # 沙箱化 Python 执行
│   └── report.py          # 训练报告生成
├── scripts/               # 入口脚本（9 个 Python 文件）
│   ├── tok_train.py       # 分词器训练
│   ├── tok_eval.py        # 分词器评估
│   ├── base_train.py      # 基座模型预训练
│   ├── base_eval.py       # 基座模型评估
│   ├── chat_sft.py        # 监督微调
│   ├── chat_rl.py         # 强化学习
│   ├── chat_eval.py       # 对话模型评估
│   ├── chat_cli.py        # CLI 聊天
│   └── chat_web.py        # Web 聊天服务器
├── tasks/                 # 评估/训练任务定义（8 个文件）
│   ├── common.py          # 任务基类和工具
│   ├── arc.py             # ARC 数据集
│   ├── mmlu.py            # MMLU 数据集
│   ├── gsm8k.py           # GSM8K 数据集
│   ├── humaneval.py       # HumanEval 数据集
│   ├── smoltalk.py        # SmolTalk 对话数据
│   ├── spellingbee.py     # 拼写/计数任务
│   └── customjson.py      # 自定义 JSONL 数据
├── tests/                 # 测试（2 个文件）
│   ├── test_engine.py     # 推理引擎测试
│   └── test_attention_fallback.py # FA3/SDPA 回退测试
├── dev/                   # 开发工具（2 个文件）
│   ├── gen_synthetic_data.py  # 合成数据生成
│   └── repackage_data_reference.py # 数据重打包（参考）
├── runs/                  # Shell 脚本（4 个）
│   ├── speedrun.sh        # GPT-2 速度竞赛脚本
│   ├── runcpu.sh          # CPU/Mac 演示
│   ├── scaling_laws.sh    # 缩放定律扫描
│   └── miniseries.sh      # 计算最优模型族训练
├── doc/                   # 文档
└── pyproject.toml         # 项目配置
```

### 3.2 模块依赖图

```
                    ┌──────────────┐
                    │  common.py   │ (底层工具：dtype, DDP, 日志)
                    └──────┬───────┘
           ┌───────────────┼───────────────────────┐
           ▼               ▼                       ▼
    ┌──────────┐    ┌──────────────┐      ┌──────────────┐
    │  gpt.py  │    │ tokenizer.py │      │ flash_attn.py│
    └────┬─────┘    └──────┬───────┘      └──────────────┘
         │                 │
    ┌────┴─────┐    ┌──────┴───────┐
    │ optim.py │    │ dataset.py   │
    │  fp8.py  │    │ dataloader.py│
    └──────────┘    └──────────────┘
                         │
    ┌────────────────────┼────────────────────┐
    ▼                    ▼                    ▼
┌──────────┐    ┌────────────────┐   ┌──────────────┐
│engine.py │    │checkpoint_mgr  │   │ core_eval.py │
└──────────┘    └────────────────┘   │ loss_eval.py │
                                     │ execution.py │
                                     │  report.py   │
                                     └──────────────┘
```

### 3.3 数据流

```
HuggingFace (ClimbMix-400B)
    │
    ▼
dataset.py (下载 + 遍历 Parquet 分片)
    │
    ▼
tok_train.py (训练 BPE 分词器) → tokenizer/
    │
    ▼
dataloader.py (BOS 对齐最佳适配打包) → base_train.py (预训练)
    │
    ▼
base_checkpoints/d{N}/ → chat_sft.py (SFT 微调)
    │
    ▼
chatsft_checkpoints/d{N}/ → chat_rl.py (RL 训练，可选)
    │
    ├──→ chat_cli.py (命令行聊天)
    ├──→ chat_web.py (Web 聊天服务器)
    └──→ chat_eval.py (模型评估)
```

---

## 四、核心实现详解

### 4.1 GPT 模型初始化和权重初始化

`GPT.__init__()` 在 meta 设备上运行（只分配形状/dtype，不分配数据），实际数据通过 `init_weights()` 初始化：

| 参数 | 初始化方式 | 标准差/范围 |
|------|-----------|------------|
| wte (词嵌入) | 正态分布 N(0, 0.8²) | 0.8 |
| lm_head | 正态分布 N(0, 0.001²) | 0.001 |
| attn.c_q, c_k, c_v | 均匀分布 | [-s, s] where s = √3/√(n_embd) |
| attn.c_proj | 零初始化 | 0 |
| mlp.c_fc | 均匀分布 × 0.4 | [-s×0.4, s×0.4] |
| mlp.c_proj | 零初始化 | 0 |
| value_embeds | 均匀分布 | [-s, s] |
| ve_gate | 均匀分布 [0, 0.02] | 小正值 |
| smear_gate | 均匀分布 [0, 0.02] | 小正值 |
| smear_lambda | 零初始化 | 0 |
| backout_lambda | 常量 0.2 | 0.2 |
| resid_lambdas | 1.15 → 1.05 线性衰减 | - |
| x0_lambdas | 0.20 → 0.05 线性衰减 | - |

### 4.2 滑动窗口注意力实现

```python
def _compute_window_sizes(self, config):
    # 长窗口 = 完整上下文
    long_window = config.sequence_len
    # 短窗口 = ceil(seq_len/4/128)*128 (向上取整到 FA3 tile 大小)
    short_window = -(-long_window // 4 // 128) * 128
    # 按模式字符串平铺到各层
    for layer_idx in range(config.n_layer):
        char = pattern[layer_idx % len(pattern)]
        window_sizes.append((long_window, 0) if char == 'L' else (short_window, 0))
    # 最后一层始终全上下文
    window_sizes[-1] = (long_window, 0)
```

### 4.3 FLOPs 估计

```python
def estimate_flops(self):
    # 每个权重参数：前向 2 FLOPs + 反向 4 FLOPs = 6 FLOPs
    nparams = sum(p.numel() for p in self.parameters())
    # 排除非矩阵乘参数（嵌入层 + 标量参数）
    nparams_exclude = wte + value_embeds + 标量参数
    # 注意力 FLOPs: 每层 12 * h * q * effective_seq_len
    for window_size in self.window_sizes:
        effective_seq = min(window, t) if window > 0 else t
        attn_flops += 12 * h * q * effective_seq
    # 总 FLOPs/Token = 6 * (总参数 - 非矩阵参数) + 注意力 FLOPs
    num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
```

### 4.4 FP8 线性层实现

核心是 `_Float8Matmul` 自定义 autograd 函数：

```python
class _Float8Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_2d, weight):
        # 量化为 e4m3（精度更高）
        input_fp8, input_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)
        ctx.save_for_backward(input_fp8, input_inv, weight_fp8, weight_inv)
        # 通过 cuBLAS FP8 内核计算
        return torch._scaled_mm(input_fp8, weight_fp8.t(), ...)

    @staticmethod
    def backward(ctx, grad_output):
        # GEMM 1: grad_input = grad_output @ weight
        go_fp8 = _to_fp8(grad_output, torch.float8_e5m2)  # e5m2 范围更广
        grad_input = torch._scaled_mm(go_fp8, w_col, ...)

        # GEMM 2: grad_weight = grad_output.T @ input
        go_T = go_fp8.t().contiguous()
        grad_weight = torch._scaled_mm(go_T, in_col, ...)
        return grad_input, grad_weight
```

> 注意：与 torchao 的 2000 行实现相比，nanochat 的实现只有 ~150 行，因为它只需要"张量级"缩放（一个标量缩放整个张量），而不需要逐行缩放、FSDP float8 all-gather、DTensor 等通用性功能。

### 4.5 Flash Attention 统一接口

```python
# 自动检测：Hopper GPU (sm90) → FA3，其他 → SDPA 回退
_fa3 = _load_flash_attention_3()  # 尝试加载 kernels.get_kernel('varunneal/flash-attention-3')

def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    if USE_FA3:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    # SDPA 回退：转置为 (B, H, T, D) 布局
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=...)
    return y.transpose(1, 2)  # 回到 (B, T, H, D)
```

### 4.6 工具使用系统

工具使用在 `Engine.generate()` 中实现，是一个基于特殊 token 的状态机：

```
初始状态 ──[<|python_start|>]──→ 在 Python 代码块中
    │                              │ (累积表达式 token)
    │                              │
    │         ──[<|python_end|>]──→ 执行 Python 代码
    │                              │ (使用 use_calculator())
    │                              │
    │         ←── 强制注入结果 ─── [<|output_start|>] + 结果 + [<|output_end|>]
    │
    ──[<|assistant_end|>]──→ 行完成
```

`use_calculator()` 支持两种操作：
1. **纯数学表达式**（如 `3*4+5`）：直接 `eval()` 计算
2. **字符串计数**（如 `'hello'.count('l')`）：支持 `.count()` 方法

安全措施：
- 禁用危险模式（`__`, `import`, `exec`, `eval`, `open` 等）
- 超时保护（默认 3 秒）
- 禁用幂运算符（`**`）

### 4.7 数据集与基准

#### 预训练数据
- **ClimbMix-400B**：NVIDIA 的高质量预训练数据集
- 以 GPT-2 tokenizer 分词后存储为 Parquet 分片
- 每个分片约 250M 字符，ZSTD 压缩后 ~100MB
- 共 6542 个训练分片 + 1 个验证分片
- 运行 `python -m nanochat.dataset -n 170` 下载足够训练 GPT-2 级别的数据

#### SFT 数据混合
| 组件 | 大小 | 目的 |
|------|------|------|
| SmolTalk | 460K 行 | 通用对话 |
| 身份对话 | 1000 行 × 2 | 模型自我认知 |
| MMLU | 100K 行 × 3 轮 | 多选题能力 |
| GSM8K | 8K 行 × 4 轮 | 数学和工具使用 |
| SimpleSpelling | 200K 行 | 拼写能力 |
| SpellingBee | 80K 行 | 字母计数能力 |

---

## 五、执行流程

### 5.1 完整流水线 (speedrun.sh)

```
步骤 1: 下载数据集
  python -m nanochat.dataset -n 170
  → 下载约 170 个 ClimbMix-400B 分片到 ~/.cache/nanochat/base_data_climbmix/

步骤 2: 训练分词器
  python -m scripts.tok_train
  → 训练 32K 词汇量的 BPE 分词器
  → 保存到 ~/.cache/nanochat/tokenizer/
  → 同时计算 token_bytes.pt 用于 BPB 评估

步骤 3: 预训练基座模型
  torchrun --nproc_per_node=8 -m scripts.base_train --depth=24 --fp8
  → 训练 d24 模型（24 层 Transformer）
  → 使用 FP8 加速，FA3 注意力
  → 每 2000 步评估 CORE 指标
  → 保存检查点到 ~/.cache/nanochat/base_checkpoints/d24/

步骤 4: 监督微调 (SFT)
  torchrun --nproc_per_node=8 -m scripts.chat_sft
  → 在对话数据上微调基座模型
  → 仅对助手 token 计算损失
  → 保存到 ~/.cache/nanochat/chatsft_checkpoints/d24/

步骤 5: 强化学习 (RL，可选)
  torchrun --nproc_per_node=8 -m scripts.chat_rl
  → 在 GSM8K 上进行 GRPO/REINFORCE 训练
  → 使用 DAPO 风格的 token 级归一化
  → 保存到 ~/.cache/nanochat/chatrl_checkpoints/d24/

步骤 6: 最终评估
  python -m scripts.chat_eval -i sft
  → 评估 ChatCORE 指标
  → 生成训练报告
```

### 5.2 预训练训练循环详解

```python
while True:
    # 1. 评估阶段（周期性）
    if step % eval_every == 0:
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)

    if step % core_metric_every == 0:
        results = evaluate_core(orig_model, tokenizer, device)

    if step % sample_every == 0:
        # 生成文本样本

    # 2. 保存检查点
    if last_step or step % save_every == 0:
        save_checkpoint(...)

    if last_step:
        break

    # 3. 单步训练
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)            # 前向传播
        loss = loss / grad_accum_steps # 归一化损失
        loss.backward()                # 反向传播
        x, y = next(train_loader)      # 预取下一批数据

    # 4. 优化器步骤
    optimizer.step()
    model.zero_grad()
```

### 5.3 Web 聊天服务器架构

```
                    FastAPI (chat_web.py)
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
        GET /       POST /chat/    GET /health
        (UI)        completions    GET /stats
                        │
                   WorkerPool
                   ┌───┴───┐
              ┌────┴──┐  ┌──┴────┐
              │ GPU 0 │  │ GPU 1 │  ...
              │ Model │  │ Model │
              └───────┘  └───────┘
```

- **WorkerPool**：每个 GPU 加载一份完整的模型副本
- **异步队列**：`asyncio.Queue` 管理可用 worker
- **流式响应**：SSE (Server-Sent Events) 流式返回 token
- **滥用防护**：
  - 最多 500 条消息/请求
  - 单条消息最多 8000 字符
  - 总对话最多 32000 字符
  - Temperature 限制 [0.0, 2.0]
  - Top-k 限制 [0, 200]
  - Max tokens 限制 [1, 4096]

---

## 六、关键设计决策

### 6.1 精度管理

nanochat 不使用 `torch.amp.autocast`，而是通过全局 `COMPUTE_DTYPE` 管理精度：

```python
# 自动检测最合适的计算精度
if CUDA SM >= 80 (Ampere+):
    COMPUTE_DTYPE = torch.bfloat16  # BF16 张量核心
else:
    COMPUTE_DTYPE = torch.float32   # 老 GPU 或 CPU

# Linear 层在 forward 时将权重重铸为输入 dtype
class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))
```

### 6.2 无偏置设计

遵循现代 Transformer 设计，所有 Linear 层均不使用 bias：
- 减少参数量
- 简化优化器逻辑
- RMSNorm 也无学习参数

### 6.3 Meta 设备初始化

模型先在 `meta` 设备上构建（不分配实际内存），然后：
1. `to_empty(device)`：在目标设备上分配未初始化内存
2. `init_weights()`：就地初始化所有权重
3. 如需加载检查点：`load_state_dict(..., assign=True)` 覆盖

这种模式避免了两次内存分配（先初始化再加载检查点）。

### 6.4 垃圾回收管理

训练过程中手动管理 Python GC 以避免 ~500ms 的暂停：

```python
if first_step:
    gc.collect()  # 手动清理初始化产生的垃圾
    gc.freeze()   # 冻结当前所有对象，排除在 GC 之外
    gc.disable()  # 禁用 GC
elif step % 5000 == 0:
    gc.collect()  # 每 5000 步手动回收一次
```

### 6.5 评估指标体系

| 指标 | 阶段 | 描述 |
|------|------|------|
| BPB (Bits Per Byte) | 预训练/SFT | 词汇量无关的损失，主指标 |
| CORE | 基座模型 | 上下文学习能力，GPT-2 基准 0.256525 |
| ChatCORE | 对话模型 | 6 个任务的平均中心化准确率 |
| Pass@k | RL | 生成 k 个样本至少一个正确的比例 |

### 6.6 对数软封顶 (Logit Softcapping)

```python
softcap = 15
logits = softcap * torch.tanh(logits / softcap)
```

作用：将 logits 平滑限制在 [-15, 15]，防止模型输出极端概率，抑制过度自信。

---

## 附录 A：模型参数初始化详细表

| 参数组 | 初始化 | 说明 |
|--------|--------|------|
| `wte.weight` | N(0, 0.8²) | 词嵌入 |
| `lm_head.weight` | N(0, 0.001²) | 输出投影（极小的初始值） |
| `attn.c_q.weight` | U[-s, s], s=√3/√d | 均匀分布，标准差 = 1/√d |
| `attn.c_k.weight` | U[-s, s] | 同上 |
| `attn.c_v.weight` | U[-s, s] | 同上 |
| `attn.c_proj.weight` | 0 | 零初始化投影 |
| `mlp.c_fc.weight` | U[-0.4s, 0.4s] | 0.4 倍标准缩放的 MLP 输入 |
| `mlp.c_proj.weight` | 0 | 零初始化投影 |
| `value_embeds.weight` | U[-s, s] | 与 c_v 相同 |
| `ve_gate.weight` | U[0, 0.02] | 小正初始值，门控接近中性 |
| `smear_gate.weight` | U[0, 0.02] | 小正初始值 |
| `smear_lambda` | 0 | 从零开始学习 |
| `backout_lambda` | 0.2 | 固定 0.2 初始值 |
| `resid_lambdas[i]` | 1.15 - 0.10×i/(n-1) | 浅层强残差，深层弱残差 |
| `x0_lambdas[i]` | 0.20 - 0.15×i/(n-1) | 浅层更多初始嵌入混合 |
