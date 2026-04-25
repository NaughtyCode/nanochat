"""
GPT 模型（重写版，大幅简化）
主要特性：
- 旋转位置编码（RoPE，无传统位置嵌入）
- QK 归一化（QK Norm）
- token embedding 与 lm_head 权重不共享（untied weights）
- MLP 中使用 ReLU² 激活函数
- token embedding 之后进行归一化
- RMSNorm 中无可学习参数
- 线性层无偏置（bias=False）
- 分组查询注意力（GQA）支持更高效的推理
- Flash Attention 3 集成（Hopper GPU），其他硬件自动回退到 SDPA
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    """GPT 模型配置"""
    sequence_len: int = 2048          # 序列长度（上下文窗口）
    vocab_size: int = 32768           # 词汇表大小
    n_layer: int = 12                 # Transformer 层数
    n_head: int = 6                   # 查询头数量（Q heads）
    n_kv_head: int = 6                # KV 头数量（GQA，可少于 n_head）
    n_embd: int = 768                 # 嵌入维度（模型宽度）
    # 滑动窗口注意力模式字符串，循环平铺到各层。最后一层总是 L（全上下文）。
    # 字符含义: L=long（全上下文），S=short（四分之一上下文）
    # 示例: "L"=全部全上下文, "SL"=交替模式, "SSL"=两层短一层长
    window_pattern: str = "SSSL"


def norm(x):
    """RMS 归一化（无偏置，无缩放参数），在 bf16 精度下运行"""
    return F.rms_norm(x, (x.size(-1),))

class Linear(nn.Linear):
    """自定义线性层：forward 时将权重转换为输入的 dtype。
    替代 torch.amp.autocast：主权重保持 fp32 以获得优化器精度，
    但矩阵乘法在激活的 dtype（通常为 bf16）中运行。"""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """判断某层是否应包含 Value Embedding（交替模式，最后一层始终包含）。
    Value Embedding 是 ResFormer 风格的改进，只在部分层添加以节省参数。"""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    """应用旋转位置编码（RoPE）。
    将最后一维分成两半，每一对 (x_i, x_{i+d/2}) 在 2D 空间中旋转。
    旋转矩阵: [cos, -sin; sin, cos]
    这使注意力分数能够编码 token 之间的相对位置信息。"""
    assert x.ndim == 4  # 多头注意力: (B, T, H, D)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # 将最后一维分成两半
    y1 = x1 * cos + x2 * sin          # 旋转每对维度
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    """因果自注意力层。
    使用 FA3（Hopper GPU）或 PyTorch SDPA（回退方案）实现高效的 Flash Attention。
    支持 GQA（分组查询注意力）、滑动窗口注意力、QK 归一化、Value Embedding。"""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head          # 查询头数
        self.n_kv_head = config.n_kv_head     # KV 头数（GQA：可少于 n_head）
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head  # 每个头的维度
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        # 线性投影层（无偏置）
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        # Value Embedding 门控（ResFormer 风格）：仅在有 VE 的层创建
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        """前向传播。
        ve: Value Embedding（ResFormer），可为 None
        cos_sin: (cos, sin) 旋转位置编码
        window_size: (left, right) 滑动窗口大小
        kv_cache: KV 缓存（推理时使用，训练时为 None）"""
        B, T, C = x.size()

        # 将输入投影为 Q、K、V
        # 形状: (B, T, H, D) — FA3 的原生布局，无需转置！
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value 残差（ResFormer）：使用依赖输入的逐头门控将 value embedding 混合进来
        # gate 范围 (0, 3)，由输入的前 ve_gate_channels 维经 sigmoid 后乘以 3 得到
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # 对 Q 和 K 应用旋转位置编码（RoPE）以编码相对位置信息
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)    # QK 归一化：稳定训练，防止注意力分数过大
        q = q * 1.2                # 更尖锐的注意力分布（将缩放分散到 Q 和 K 之间）
        k = k * 1.2

        # Flash Attention（Hopper GPU 使用 FA3，其他硬件自动回退到 SDPA）
        if kv_cache is None:
            # 训练路径：因果注意力 + 可选的滑动窗口
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # 推理路径：使用 flash_attn_with_kvcache 管理 KV 缓存
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # 最后一层处理完后推进缓存位置
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # 重新组合所有头并投影回残差流
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """前馈网络（MLP）。
    使用 ReLU² 激活函数（Gated ReLU 的简化版）：
    - 先 ReLU 过滤负值，再平方 → 更平滑的梯度流
    - 扩展比为 4（标准 Transformer 设计）"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()    # ReLU² 激活
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Transformer 块：Pre-norm 残差结构。
    每个 Block = CausalSelfAttention + MLP，两次使用 RMSNorm 预归一化。"""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)  # 注意力子层
        x = x + self.mlp(norm(x))                                       # MLP 子层
        return x


class GPT(nn.Module):
    """GPT 模型主体。
    包含 Transformer 层、嵌入、输出头以及各种辅助机制
    （Smear、Backout、Value Embeddings、Residual/X0 lambdas）。"""
    def __init__(self, config, pad_vocab_size_to=64):
        """
        注意：此 __init__ 在 meta device 上下文中运行（因为使用了 torch.device('meta')）。
        因此此处的所有计算只是形状和 dtype 占位，不包含实际数据。
        真正的参数初始化在 init_weights() 中完成。

        pad_vocab_size_to: 将 vocab_size 填充到此值的倍数，以优化 DDP 和 Tensor Core 效率。
        """
        super().__init__()
        self.config = config
        # 计算每层的滑动窗口大小
        self.window_sizes = self._compute_window_sizes(config)
        # 将 vocab 填充到 pad_vocab_size_to 的倍数以提高效率（DDP、Tensor Core 偏好对齐的内存）
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),  # token 嵌入
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),  # Transformer 层
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)  # 输出投影（与 wte 权重不共享）
        # 逐层可学习的缩放因子（灵感来自 modded-nanogpt）
        # resid_lambdas: 缩放每层的残差流（初值 1.0 = 中性，深层通常更大）
        # x0_lambdas: 将初始嵌入混合回每层的残差流（初值 0.0 = 禁用，越早层越大）
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))    # 占位初始值，真实值在 init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))      # 占位初始值，真实值在 init_weights()
        # Smear：将上一个 token 的嵌入混合到当前 token（廉价地引入类似 bigram 的信息）
        self.smear_gate = Linear(24, 1, bias=False)                      # 门控：从输入的前 24 维计算混合权重
        self.smear_lambda = nn.Parameter(torch.zeros(1))                 # 全局缩放因子
        # Backout：在最终归一化前减去缓存的中间层残差，去除低级特征，让 lm_head 关注高层语义
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value Embeddings（ResFormer 风格）：交替层包含，最后一层始终包含
        # 将 token 的 value 信息直接注入注意力计算，类似于残差连接
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # 预计算旋转位置编码（RoPE）：序列长度 ×10 以确保足够长的序列
        # 这些编码在内存中很便宜，10 倍过度计算足以覆盖极端情况
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)  # persistent=False 表示不保存到 checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """完整模型权重初始化（所有参数在一个函数中初始化，保证清晰）。
        初始化策略：
        - wte（嵌入）:      正态分布, std=0.8
        - lm_head:           正态分布, std=0.001
        - attn.c_q/c_k/c_v:  均匀分布, std=1/√(n_embd)（使用 sqrt(3) 乘子使均匀分布与正态分布同方差）
        - attn.c_proj:       零初始化
        - mlp.c_fc:          均匀分布, std=1/√(n_embd)×0.4（0.4 倍缩放）
        - mlp.c_proj:        零初始化
        - resid_lambdas:     从 1.15 线性衰减到 1.05（早期的残差更强）
        - x0_lambdas:        从 0.20 线性衰减到 0.05（早期层更多原始嵌入混合）
        - smear/backout:     零/小正数初始化
        - value_embeds:      均匀分布（与 c_v 相同）
        - ve_gate:           小正数均匀分布（门控从接近中性开始）
        零初始化投影层的原因：残差流在初始时主要由残差连接传递，
        零初始化确保训练初期模型表现为恒等映射，有利于训练稳定性。
        """

        # 嵌入和反嵌入层
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer 块：使用均匀分布以避免正态分布的离群值
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5  # sqrt(3) 乘子使均匀分布与正态分布具有相同的标准差
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)           # 投影层零初始化
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)            # 投影层零初始化

        # 逐层缩放因子
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/Backout 缩放因子和门控
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value Embeddings（与 c_v 相同的初始化方式）
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # VE 门控权重用小正数初始化（门控从接近中性开始）
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # 旋转位置编码
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # 将嵌入转换为 COMPUTE_DTYPE（节省内存）。
        # 例外：fp16 需要 fp32 嵌入，因为 GradScaler 无法反缩放 fp16 梯度。
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        """预计算旋转位置编码（RoPE）。
        base: 频率基数（100K 是较新的常见选择，有助于长序列外推）
        返回 (cos, sin) 形状为 (1, seq_len, 1, head_dim/2)，便于后续广播。"""
        if device is None:
            device = self.transformer.wte.weight.device
        # 步进通道：每隔一个处理一对旋转维度
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # 步进时间步
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # 计算每个 (时间, 通道) 对的旋转频率
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]  # 添加 batch 和 head 维度用于广播
        return cos, sin

    def _compute_window_sizes(self, config):
        """计算每层的滑动窗口注意力窗口大小。
        返回 (left, right) 元组列表：
        - left: 向前看多少 token（-1 = 无限制，即全上下文）
        - right: 向后看多少 token（0 = 因果注意力）
        模式字符串循环平铺到各层，最后一层始终为全上下文。
        S 的窗口大小向上取整到 128 的倍数（FA3 tile 大小对齐要求）。"""
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # 向上取整到 FA3 tile 大小
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # 将模式循环平铺到各层
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # 最后一层始终为全上下文
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        """返回模型所在的设备"""
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """估算模型每个 token 的 FLOPs（前向 + 反向）。
        公式：
        - 每个矩阵乘法权重参数贡献 6 FLOPs：
          前向 = 2（乘法和累加），反向 = 4（2×前向）→ 2+4=6
        - 注意力：12 × h × q × effective_seq_len
          其中 effective_seq_len 考虑滑动窗口，每层可能不同
        与 Chinchilla 论文的公式有约 1% 差异：
        - Chinchilla 将嵌入层算作 FLOPs → 我们忽略（嵌入只是查表）
        - Chinchilla 将 softmax 中的 exp/sum/division 算作 FLOPs → 量级很小，忽略
        """
        nparams = sum(p.numel() for p in self.parameters())
        # 排除非矩阵乘法参数：嵌入和逐层缩放因子
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # 对各层注意力 FLOPs 求和，考虑滑动窗口
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]   # (left, right) 元组，取 left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """返回详细的参数计数，用于缩放定律分析。
        不同论文使用不同惯例：
        - Kaplan et al. 排除嵌入参数
        - Chinchilla 包含所有参数
        返回每个参数组计数，以便下游分析可以测试哪种组合给出最干净的缩放定律。"""
        # 分别统计每组参数（与 setup_optimizer 的分组一致）
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        """设置优化器：将参数分组并分配给 Muon 或 AdamW。
        - Muon：Transformer 块中的 2D 矩阵参数（Q、K、V、投影、MLP 权重）
        - AdamW：嵌入层、lm_head、标量参数（resid_lambdas、x0_lambdas、smear/backout）
        学习率按 ∝1/√(dmodel) 缩放（以 768 维模型为参考调优）。
        """
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # 将所有参数分为不同的组
        matrix_params = list(self.transformer.h.parameters())         # Muon 管理
        value_embeds_params = list(self.value_embeds.parameters())    # AdamW 管理
        embedding_params = list(self.transformer.wte.parameters())    # AdamW 管理
        lm_head_params = list(self.lm_head.parameters())              # AdamW 管理
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # 对 AdamW 参数按 ∝1/√(dmodel) 缩放学习率（针对 768 维模型调优）
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # 构建参数组（所有必要字段显式列出）
        param_groups = [
            # AdamW 组（嵌入、lm_head、标量）
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # x0 使用更高的 beta1
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Muon 组（矩阵参数，按形状分组以便堆叠处理）
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW  # 分布式训练使用 DistMuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]  # 记录初始学习率，用于 scheduler
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        """GPT 模型前向传播。
        idx: 输入 token IDs，形状 (B, T)
        targets: 目标 token IDs（训练时使用），形状 (B, T)
        kv_cache: KV 缓存（推理时使用），训练时为 None
        loss_reduction: 损失归约方式（'mean'、'sum' 或 'none'）
        """
        B, T = idx.size()

        # 获取当前序列长度的旋转编码
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # 如果有 KV 缓存，需要将旋转编码偏移到缓存中的当前位���
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]  # 截取当前序列长度

        # Token 嵌入
        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)  # 确保激活在计算 dtype 中（通常无操作，但对 fp16 路径有作用）
        x = norm(x)              # 嵌入后归一化

        # Smear 机制：将上一个 token 的嵌入混合到当前位置（廉价地提供类似 bigram 的信息）
        if kv_cache is None:
            # 训练 / 朴素生成：完整序列可用，使用快速切片
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV 缓存推理：从缓存读上一个嵌入，存储当前嵌入供下一步使用
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]  # 保存最后一个 token 的嵌入
            if T > 1:
                # Prefill：对位置 1+ 应用 smear，与训练相同
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode：单个 token，使用缓存的上一嵌入
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Transformer 主体前向传播
        x0 = x  # 保存初始归一化嵌入，用于 x0 残差
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # 在中间层缓存残差（用于 backout）
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            # 逐层残差缩放：resid_lambda 缩放残差流，x0_lambda 混合初始嵌入
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            # 获取当前层的 Value Embedding（如果该层有的话）
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x  # 缓存中间层残差

        # Backout：减去中间层残差，去除低级特征，让 lm_head 关注高层语义
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # lm_head：计算 logits
        softcap = 15  # 平滑地将 logits 限制在 [-softcap, softcap] 范围内
        logits = self.lm_head(x)                              # (B, T, padded_vocab_size) — 非常大的张量
        logits = logits[..., :self.config.vocab_size]         # 切片去除填充
        logits = logits.float()                               # 转为 fp32 进行 softcap 和损失计算
        logits = softcap * torch.tanh(logits / softcap)       # 软截断 logits

        if targets is not None:
            # 训练模式：计算并返回损失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # 推理模式：直接返回 logits
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """朴素自回归流式推理（简单实现，batch_size=1）。
        tokens: Python 列表形式的输入 token IDs
        max_tokens: 最大生成 token 数
        temperature: 采样温度（0 = 贪心解码）
        top_k: Top-K 采样（None = 禁用）
        seed: 随机种子
        生成器：逐个 yield 生成的 token ID（int）。
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # 添加 batch 维度
        for _ in range(max_tokens):
            logits = self.forward(ids)               # (B, T, vocab_size)
            logits = logits[:, -1, :]                # 只取最后一个位置的 logits (B, vocab_size)
            if top_k is not None and top_k > 0:      # Top-K 过滤
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)  # 贪心解码
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
