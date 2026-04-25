"""
Flash Attention 统一接口测试 — 验证 FA3（Flash Attention 3）与 SDPA 产生相同结果。

运行：python -m pytest tests/test_attention_fallback.py -v -s

测试结构说明：
    测试分为三个类，受 dtype/device 约束：

    1. TestFA3VsSDPA — 对比测试，在相同输入上分别运行 FA3 和 SDPA，
       验证两者输出一致。需要 Hopper GPU（FA3 仅支持 sm90+），
       使用 bfloat16（FA3 不支持 float32）。

    2. TestSDPAOnly — 仅测试 SDPA 回退路径，可在任何设备（CUDA/CPU/MPS）
       上运行，使用对应设备支持的 dtype。

    3. TestOverrideMechanism — 测试实现选择机制（override/auto 切换）。

覆盖的测试场景：
    - 基本因果注意力、全上下文、滑动窗口
    - GQA（Group Query Attention）：KV 头数少于 Q 头数
    - 更接近真实模型的大维度
    - KV Cache 场景：prefill（多 token 写入空 cache）、单 token decode
    - 滑动窗口 decode（曾发现 SDPA 在单 token 时忽略 window_size 的 bug）
    - 反向传播梯度一致性
"""
import torch
import pytest
import nanochat.flash_attention as fa_module
from nanochat.flash_attention import flash_attn, HAS_FA3
from nanochat.engine import KVCache


def set_impl(impl):
    """Set the implementation override ('fa3', 'sdpa', or None for auto) and re-resolve USE_FA3."""
    fa_module._override_impl = impl
    fa_module.USE_FA3 = fa_module._resolve_use_fa3()


def run_both_impls(fn):
    """Run a function with both FA3 and SDPA, return both outputs."""
    set_impl('fa3')
    out_fa3 = fn()
    set_impl('sdpa')
    out_sdpa = fn()
    set_impl(None)  # reset
    return out_fa3, out_sdpa


def assert_close(t1, t2, name, atol=1e-2, rtol=1e-2):
    """Assert two tensors are close, with helpful error message."""
    max_diff = (t1 - t2).abs().max().item()
    mean_diff = (t1 - t2).abs().mean().item()
    assert torch.allclose(t1, t2, atol=atol, rtol=rtol), \
        f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
    return max_diff, mean_diff


# =============================================================================
# FA3 vs SDPA comparison tests (require Hopper GPU)
# =============================================================================
@pytest.mark.skipif(not HAS_FA3, reason="FA3 required to compare implementations")
class TestFA3VsSDPA:
    """对比 FA3 与 SDPA 产生相同结果。需要 Hopper GPU (sm90+)。"""

    DEVICE = "cuda"
    DTYPE = torch.bfloat16

    def test_basic_causal(self):
        """基本因果注意力测试。"""
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "basic_causal")
        print(f"basic_causal: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_full_context(self):
        """全上下文注意力（window_size=-1，无窗口限制）。"""
        B, T, H, D = 2, 128, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "full_context")
        print(f"full_context: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_sliding_window(self):
        """滑动窗口注意力 — 每个 token 只能关注前 window 个 token。"""
        B, T, H, D = 2, 128, 4, 32
        window = 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(window, 0))

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "sliding_window")
        print(f"sliding_window: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_gqa(self):
        """分组查询注意力（GQA）：KV 头数（2）少于 Q 头数（8）。"""
        B, T, D = 2, 64, 32
        n_heads = 8
        n_kv_heads = 2

        q = torch.randn(B, T, n_heads, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, n_kv_heads, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, n_kv_heads, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "gqa")
        print(f"gqa: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_larger_model(self):
        """更接近真实模型的维度（12 头、64 维、256 序列长）。"""
        B, T, H, D = 4, 256, 12, 64
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "larger_model")
        print(f"larger_model: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_kvcache_prefill(self):
        """测试 prefill：将多个 token 一次性写入空 KV Cache。"""
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 16

        q = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            k_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            v_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            cache_seqlens = torch.zeros(B, dtype=torch.int32, device=self.DEVICE)
            return flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache, k=k, v=v,
                cache_seqlens=cache_seqlens,
                causal=True, window_size=(T_max, 0)
            )

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "prefill")
        print(f"prefill: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_kvcache_single_token(self):
        """测试单 token 逐 token 生成（cache 中已有内容）。"""
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 16

        k_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            k_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            v_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            k_cache[:, :T_prefill, :, :] = k_init
            v_cache[:, :T_prefill, :, :] = v_init
            cache_seqlens = torch.full((B,), T_prefill, dtype=torch.int32, device=self.DEVICE)
            return flash_attn.flash_attn_with_kvcache(
                q_single, k_cache, v_cache, k=k_single, v=v_single,
                cache_seqlens=cache_seqlens,
                causal=True, window_size=(T_max, 0)
            )

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "single_token")
        print(f"single_token: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_kvcache_single_token_sliding_window(self):
        """
        回归测试：单 token 解码时滑动窗口正确性。

        曾发现 bug：SDPA 在 Tq=1 的 decode 阶段忽略了 window_size 参数，
        导致单 token 时仍关注所有缓存 token（而非仅关注窗口内的）。
        当 window=8 远小于 T_prefill=32 时，FA3 和 SDPA 的输出应有显著差异。
        """
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 32  # Enough tokens to exceed window
        window = 8      # Window SMALLER than cache size

        k_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            k_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            v_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            k_cache[:, :T_prefill, :, :] = k_init
            v_cache[:, :T_prefill, :, :] = v_init
            cache_seqlens = torch.full((B,), T_prefill, dtype=torch.int32, device=self.DEVICE)
            return flash_attn.flash_attn_with_kvcache(
                q_single, k_cache, v_cache, k=k_single, v=v_single,
                cache_seqlens=cache_seqlens,
                causal=True, window_size=(window, 0)  # window=8 < Tk=33
            )

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "single_token_sliding_window")
        print(f"single_token_sliding_window: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_backward_gradients_match(self):
        """验证 FA3 与 SDPA 反向传播的梯度一致（较高容差 0.05）。"""
        B, T, H, D = 2, 32, 4, 16

        q_data = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_data = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_data = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            q = q_data.clone().requires_grad_(True)
            k = k_data.clone().requires_grad_(True)
            v = v_data.clone().requires_grad_(True)
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))
            loss = y.sum()
            loss.backward()
            return y.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()

        set_impl('fa3')
        y_fa3, q_grad_fa3, k_grad_fa3, v_grad_fa3 = run()
        set_impl('sdpa')
        y_sdpa, q_grad_sdpa, k_grad_sdpa, v_grad_sdpa = run()
        set_impl(None)

        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "backward_output")
        print(f"backward_output: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        max_diff, mean_diff = assert_close(q_grad_fa3, q_grad_sdpa, "q_grad", atol=0.05, rtol=0.05)
        print(f"q_grad: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        max_diff, mean_diff = assert_close(k_grad_fa3, k_grad_sdpa, "k_grad", atol=0.05, rtol=0.05)
        print(f"k_grad: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        max_diff, mean_diff = assert_close(v_grad_fa3, v_grad_sdpa, "v_grad", atol=0.05, rtol=0.05)
        print(f"v_grad: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")


# =============================================================================
# SDPA-only tests (run on any device)
# =============================================================================
class TestSDPAOnly:
    """仅测试 SDPA 回退路径。可在任何设备上运行。"""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def test_basic_forward(self):
        """测试 SDPA 前向传播产生有效输出（无 NaN）。"""
        set_impl('sdpa')
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        assert y.shape == (B, T, H, D)
        assert not torch.isnan(y).any(), "Output contains NaN"
        set_impl(None)

    def test_backward(self):
        """测试 SDPA 反向传播：梯度存在且无 NaN。"""
        set_impl('sdpa')
        B, T, H, D = 2, 32, 4, 16
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))
        loss = y.sum()
        loss.backward()

        assert q.grad is not None, "No gradient for q"
        assert k.grad is not None, "No gradient for k"
        assert v.grad is not None, "No gradient for v"
        assert not torch.isnan(q.grad).any(), "NaN in q gradient"
        set_impl(None)

    def test_kvcache(self):
        """测试 SDPA 与 KVCache 的集成：prefill + 单 token decode。"""
        set_impl('sdpa')
        B, T_max, H, D = 2, 64, 4, 32
        n_layers = 1

        cache = KVCache(
            batch_size=B, num_heads=H, seq_len=T_max, head_dim=D,
            num_layers=n_layers, device=self.DEVICE, dtype=self.DTYPE
        )
        k_cache, v_cache = cache.get_layer_cache(0)

        # Prefill
        T_prefill = 16
        q = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y = flash_attn.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v,
            cache_seqlens=cache.cache_seqlens,
            causal=True, window_size=(T_max, 0)
        )
        cache.advance(T_prefill)

        assert y.shape == (B, T_prefill, H, D)
        assert cache.get_pos() == T_prefill

        # Generate single token
        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y_single = flash_attn.flash_attn_with_kvcache(
            q_single, k_cache, v_cache, k=k_single, v=v_single,
            cache_seqlens=cache.cache_seqlens,
            causal=True, window_size=(T_max, 0)
        )
        cache.advance(1)

        assert y_single.shape == (B, 1, H, D)
        assert cache.get_pos() == T_prefill + 1
        set_impl(None)


# =============================================================================
# Override mechanism tests
# =============================================================================
class TestOverrideMechanism:
    """测试实现选择机制：override='fa3'/'sdpa'/None(auto) 的切换。"""

    @pytest.mark.skipif(not HAS_FA3, reason="FA3 required")
    def test_override_fa3(self):
        """override='fa3' 强制使用 FA3。"""
        set_impl('fa3')
        assert fa_module.USE_FA3 == True
        set_impl(None)

    def test_override_sdpa(self):
        """override='sdpa' 强制使用 SDPA。"""
        set_impl('sdpa')
        assert fa_module.USE_FA3 == False
        set_impl(None)

    def test_override_auto(self):
        """override=None 使用自动检测（有 FA3 则用 FA3，否则 SDPA）。"""
        set_impl(None)
        assert fa_module.USE_FA3 == HAS_FA3


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        major, minor = torch.cuda.get_device_capability()
        print(f"Compute capability: {major}.{minor}")
    print(f"HAS_FA3: {HAS_FA3}")
    print()

    pytest.main([__file__, "-v", "-s"])
