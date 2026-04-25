"""
统一的 Flash Attention 接口，支持自动 FA3/SDPA 切换。

导出 `flash_attn` 模块，API 与 FA3 完全匹配，但在非 Hopper GPU（包括 Blackwell）、
MPS 和 CPU 上自动回退到 PyTorch SDPA。

使用方式（FA3 的直接替代）：
    from nanochat.flash_attention import flash_attn

    # 训练（无 KV 缓存）
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # 推理（带 KV 缓存）
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load FA3 on Hopper+ GPUs
# =============================================================================
def _load_flash_attention_3():
    """尝试加载 Flash Attention 3（仅 Hopper GPU sm90 支持）。
    Ada (sm89) 和 Blackwell (sm100) 需要 FA3 重新编译，当前使用 SDPA 回退。"""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major != 9:
            return None
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None

# Override for testing: set to 'fa3', 'sdpa', or None (auto)
_override_impl = None


def _resolve_use_fa3():
    """基于可用性、覆盖设置和 dtype 决定是否使用 FA3。
    FA3 Hopper 内核仅支持 bf16 和 fp8；fp16/fp32 必须使用 SDPA 回退。"""
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return True
    if _override_impl == 'sdpa':
        return False
    if HAS_FA3:
        from nanochat.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE == torch.bfloat16:
            return True
        return False
    return False

USE_FA3 = _resolve_use_fa3()


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """SDPA 注意力（支持滑动窗口）。
    q, k, v 为 (B, H, T, D) 格式。
    - 全上下文 + 等长：直接使用 is_causal
    - 单 token 解码：截取窗口内的 KV
    - 分块推理（Tq ≠ Tk）：构造显式 bool 掩码，同时支持滑动窗口
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """Flash Attention 训练接口（无 KV 缓存）。
    q, k, v 形状为 (B, T, H, D)。返回形状为 (B, T, H, D) 的输出张量。
    DA3 使用 (B, T, H, D) 原生布局，SDPA 需要转置为 (B, H, T, D) 再转回。"""
    if USE_FA3:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """Flash Attention 推理接口（带 KV 缓存）。
    FA3 原地更新 k_cache/v_cache。SDPA 回退方案也做到了这一点。
    - q: 查询，形状 (B, T_new, H, D)
    - k_cache, v_cache: 预分配的缓存张量，形状 (B, T_max, H_kv, D)
    - k, v: 要插入的新 key/value，形状 (B, T_new, H_kv, D)
    - cache_seqlens: 缓存中的当前位置，形状 (B,) int32
    - window_size: (left, right) 滑动窗口。 -1 表示无限制。
    返回形状为 (B, T_new, H, D) 的输出张量。"""
    if USE_FA3:
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
