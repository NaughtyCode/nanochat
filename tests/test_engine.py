"""
Engine 类的单元测试。
测试 KV Cache、生成采样、多样本推理、温度控制等核心功能。

运行方式：python -m pytest tests/test_engine.py -v
"""

import torch
from nanochat.engine import KVCache, Engine
from dataclasses import dataclass


# -----------------------------------------------------------------------------
# Mock classes for testing Engine without loading a real model

@dataclass
class MockConfig:
    """用于 Engine 测试的最小化配置，模拟 GPT 模型参数。"""
    n_kv_head: int = 4
    n_head: int = 4
    n_embd: int = 64
    n_layer: int = 2
    sequence_len: int = 128


class MockModel:
    """
    模拟模型 — forward 返回均匀分布的 logits（所有 token 等概率）。

    设计意图：使用 temperature > 0 采样时，不同样本有极高概率
    产生不同的 token。如果所有样本 token 相同，说明存在广播 bug
    （把同一个 token 复制到了所有样本行）。
    """
    def __init__(self, vocab_size=262):  # 256 bytes + 6 special tokens
        self.vocab_size = vocab_size
        self.config = MockConfig()
        self._device = torch.device("cpu")

    def get_device(self):
        return self._device

    def forward(self, ids, kv_cache=None):
        """Return uniform logits so sampling is spread across vocab."""
        B, T = ids.shape
        # With FA3, flash_attn_with_kvcache updates cache in-place and we advance position
        if kv_cache is not None:
            kv_cache.advance(T)
        # Uniform logits -> equal probability for all tokens
        logits = torch.zeros(B, T, self.vocab_size)
        return logits


class ByteTokenizer:
    """
    用于测试的简化字节级分词器。

    0-255 为原始字节，256+ 为特殊 token。
    与真实 tokenizer 的接口兼容，但无需加载模型/词表。
    """
    def __init__(self):
        # Special tokens start at 256
        self._special_tokens = {
            "<|python_start|>": 256,
            "<|python_end|>": 257,
            "<|output_start|>": 258,
            "<|output_end|>": 259,
            "<|assistant_end|>": 260,
            "<|bos|>": 261,
        }
        self._bos = 261

    def encode_special(self, s):
        return self._special_tokens[s]

    def get_bos_token_id(self):
        return self._bos

    def encode(self, s, prepend=None):
        tokens = list(s.encode("utf-8"))  # bytes 0-255
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, tokens):
        # Filter out special tokens before decoding
        byte_tokens = [t for t in tokens if t < 256]
        return bytes(byte_tokens).decode("utf-8", errors="replace")

def test_kv_cache_basic():
    """测试 KVCache 基本功能：初始化、advance、reset、get_layer_cache 视图。"""
    batch_size = 2
    num_heads = 3
    seq_len = 64
    head_dim = 5
    num_layers = 6

    kv_cache = KVCache(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        num_layers=num_layers,
        device="cpu",
        dtype=torch.float32,
    )

    # Check initial state
    assert kv_cache.get_pos() == 0
    assert kv_cache.k_cache.shape == (num_layers, batch_size, seq_len, num_heads, head_dim)
    assert kv_cache.v_cache.shape == (num_layers, batch_size, seq_len, num_heads, head_dim)

    # Test advance
    kv_cache.advance(10)
    assert kv_cache.get_pos() == 10

    kv_cache.advance(5)
    assert kv_cache.get_pos() == 15

    # Test reset
    kv_cache.reset()
    assert kv_cache.get_pos() == 0

    # Test get_layer_cache returns correct views
    k_layer0, v_layer0 = kv_cache.get_layer_cache(0)
    assert k_layer0.shape == (batch_size, seq_len, num_heads, head_dim)
    assert v_layer0.shape == (batch_size, seq_len, num_heads, head_dim)


def test_kv_cache_prefill():
    """测试 KVCache.prefill()：源 cache 的数据能正确复制到目标 cache。"""
    batch_size = 1
    num_heads = 4
    head_dim = 8
    num_layers = 2

    # Create source cache and advance it
    src_cache = KVCache(
        batch_size=batch_size, num_heads=num_heads, seq_len=32,
        head_dim=head_dim, num_layers=num_layers, device="cpu", dtype=torch.float32,
    )
    # Write some data to source cache
    src_cache.k_cache[0, 0, :16, :, :] = 1.0
    src_cache.v_cache[0, 0, :16, :, :] = 2.0
    src_cache.advance(16)

    # Create destination cache with larger seq_len
    dst_cache = KVCache(
        batch_size=batch_size, num_heads=num_heads, seq_len=64,
        head_dim=head_dim, num_layers=num_layers, device="cpu", dtype=torch.float32,
    )

    # Prefill
    dst_cache.prefill(src_cache)

    # Check position was copied
    assert dst_cache.get_pos() == 16

    # Check data was copied
    assert (dst_cache.k_cache[0, 0, :16, :, :] == 1.0).all()
    assert (dst_cache.v_cache[0, 0, :16, :, :] == 2.0).all()


def test_multi_sample_first_token_diversity():
    """
    回归测试：验证多样本生成时，每个样本的第一个 token 是独立采样的。

    早期 bug：prefill 后的第一个 token 只采样一次然后广播到所有行，
    导致所有样本起始完全相同。修复后将 prefill logits 扩展到 num_samples
    并为每行独立采样。

    在 262 词表的均匀分布下，16 个样本全部独立取到同一 token 的概率
    约为 (1/262)^15 ≈ 10^-36，因此如果所有样本 token 相同即可判定 bug 存在。
    """
    model = MockModel(vocab_size=262)
    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    # Generate 16 samples with temperature=1.0 (stochastic sampling)
    prompt_tokens = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"
    num_samples = 16

    # Collect the first generated token from each sample
    first_tokens = []
    gen = engine.generate(
        prompt_tokens,
        num_samples=num_samples,
        max_tokens=1,  # We only need the first token
        temperature=1.0,
        seed=42,
    )
    for token_column, token_masks in gen:
        first_tokens = token_column  # This is the first (and only) yield

    # With uniform distribution and 16 samples, they should NOT all be identical
    # If they are all identical, the bug exists (broadcasting instead of sampling)
    unique_tokens = set(first_tokens)
    assert len(unique_tokens) > 1, (
        f"All {num_samples} samples got the same first token ({first_tokens[0]}). "
        f"With uniform logits, this is statistically impossible (~10^-36 probability) "
        f"unless tokens are being broadcast instead of independently sampled."
    )


def test_seed_reproducibility():
    """相同 seed 必须产生完全相同的输出（确定性验证）。"""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"

    for seed in [1, 42, 123, 999]:
        r1, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        r2, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        r3, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        assert r1 == r2 == r3, "Same seed must produce identical output for the same prompt."


def test_temperature_zero_determinism():
    """temperature=0 是确定性解码，与 seed 无关。"""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]

    r1, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=1)
    r2, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=42)
    r3, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=123)
    assert r1 == r2 == r3, "Temperature=0 must result in the same output for the same prompt regardless of seed."


def test_max_tokens_respected():
    """生成 token 数不超过 max_tokens 限制。"""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]

    for max_tokens in [1, 4, 16, 64]:
        results, _ = engine.generate_batch(prompt, max_tokens=max_tokens)
        num_generated_tokens = len(results[0]) - len(prompt)
        assert num_generated_tokens <= max_tokens, f"Generated {num_generated_tokens} tokens, expected max_tokens={max_tokens} or less."


def test_num_samples_count():
    """num_samples=N 应生成恰好 N 条序列。"""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]

    for num_samples in [1, 4, 16, 64]:
        results, _ = engine.generate_batch(prompt, num_samples=num_samples, max_tokens=3)
        assert len(results) == num_samples, f"Expected {num_samples} sequences from {num_samples} samples, got {len(results)}"


def test_different_seeds_introduce_variation_when_temperature_nonzero():
    """temperature > 0 时不同 seed 应引入采样变化（统计合理性验证）。"""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"

    outputs = set()

    for seed in [1, 42, 123, 999, 1000, 1001, 1002, 1003, 1004, 1005]:
        results, _ = engine.generate_batch(
            prompt,
            temperature=1.0,
            max_tokens=5,
            seed=seed,
        )
        outputs.add(tuple(results[0]))

    # Sanity check: sampling actually introduces variation
    assert len(outputs) > 1, "All seeds produced the same output which is statistically highly improbable."
