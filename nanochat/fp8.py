"""nanochat 的轻量级 FP8 训练模块 — 仅使用张量级动态缩放。

用约 150 行代码替代 torchao 的 Float8Linear（约 2000 行）。
我们只需要"tensorwise"方案（每个张量一个标量缩放），不需要 torchao 的全部通用性
（行级缩放、FSDP float8 all-gather、DTensor、tensor subclass dispatch table 等）。

FP8 训练原理：
标准 Linear 层执行 1 次前向矩阵乘法和 2 次反向：
  forward:      output     = input      @ weight.T
  backward:     grad_input = grad_output @ weight
                grad_weight= grad_output.T @ input

FP8 训练对这三个矩阵乘法各做：
  1. 计算 scale = FP8_MAX / max(|tensor|)
  2. 量化：fp8_tensor = clamp(tensor * scale, -FP8_MAX, FP8_MAX).to(fp8)
  3. 经由 torch._scaled_mm 执行矩阵乘法（cuBLAS FP8 内核，约比 bf16 快 2 倍）
  4. 反量化：_scaled_mm 内部使用逆 scale 处理

关键洞察：torch._scaled_mm 和 float8 dtype 是 PyTorch 内置功能。
torchao 只是这些基元之上的编排层。我们可以直接调用它们。

FP8 dtype 选择（遵循标准惯例）：
  - float8_e4m3fn: 4 位指数，3 位尾数，范围 [-448, 448]
    精度更高（更多尾数位），用于输入和权重。
  - float8_e5m2:   5 位指数，2 位尾数，范围 [-57344, 57344]
    范围更宽（更多指数位），用于可能很大的梯度。

torch._scaled_mm 内存布局要求：
  - 第一个参数 (A)：必须是行优先（contiguous）
  - 第二个参数 (B)：必须是列优先（B.t().contiguous().t()）
    如果 B 通过转置 contiguous 张量获得（如 weight.t()），则已经是列优先，无需复制。

与 torchao 方法的区别：
torchao 使用"tensor subclass"架构：Float8TrainingTensor 是 torch.Tensor 的子类，
将 FP8 数据 + scale + 元数据捆绑在一起。它实现 __torch_dispatch__ 并使用 dispatch table
拦截每个 aten 操作（mm、t、reshape、clone 等），以 FP8 感知方式处理它们。
这需要约 2000 行代码，因为每个可能触及 FP8 张量的操作都需要 handler。

我们采用更简单的方法：单个 autograd.Function (_Float8Matmul)，
接受全精度输入，内部量化为 FP8，调用 _scaled_mm，返回全精度输出。
标记为 @allow_in_graph，使 torch.compile 将其视为单个不透明节点而非尝试追踪内部。

权衡：
  - torchao: compile 分解 tensor subclass，将每个操作（amax、scale、cast、_scaled_mm）
    作为独立图节点，Inductor 可将这些与周围操作融合。
  - ours: compile 看到单个不透明调用。可优化 FP8 linear 周围的一切但无法跨边界融合。

两者调用完全相同的 cuBLAS _scaled_mm 内核 — GPU 矩阵乘法相同。
差异仅在"粘合"操作（amax、scale、cast），相比矩阵乘法微不足道。
我们的版本在实际中稍快（更少的编译开销，无 tensor subclass dispatch 成本），
但在 torch.compile 下可能产生微妙不同的浮点舍入路径。eager 模式下数值完全相同。
"""

import torch
import torch.nn as nn

from nanochat.common import COMPUTE_DTYPE

# Avoid division by zero when computing scale from an all-zeros tensor
EPS = 1e-12


@torch.no_grad()
def _to_fp8(x, fp8_dtype):
    """将张量动态量化为 FP8（张量级缩放）。
    "Tensorwise" 表示为整个张量使用一个标量缩放（与逐行缩放相对）。
    Tensorwise 更快，因为 cuBLAS 直接处理缩放；逐行缩放需要 CUTLASS 内核。
    返回 (fp8_data, inverse_scale) 供 torch._scaled_mm 使用。
    """
    fp8_max = torch.finfo(fp8_dtype).max
    amax = x.float().abs().max()
    # 将 [0, amax] 映射到 [0, fp8_max]
    # 使用 float64 做除法以保证 torch.compile 与 eager 模式的数值一致性
    scale = fp8_max / amax.double().clamp(min=EPS)
    scale = scale.float()
    # 量化：缩放到 FP8 范围 → 饱和截断（clamp 防止溢出，PyTorch 默认行为是回绕而非饱和）→ 转为 FP8
    x_scaled = x.float() * scale
    x_clamped = x_scaled.clamp(-fp8_max, fp8_max)
    x_fp8 = x_clamped.to(fp8_dtype)
    # _scaled_mm 需要 scale 的*逆*（它在 matmul 中以此将 FP8 值转换回原始范围）
    inv_scale = scale.reciprocal()
    return x_fp8, inv_scale


def _to_col_major(x):
    """将 2D 张量的内存重排为列优先布局。
    torch._scaled_mm 要求第二个操作数为列优先。
    技巧：t() → contiguous() → t()。
    结果具有相同的逻辑形状，但为列优先步幅，例如 [M, N] 张量步幅变为 (1, M) 而非 (N, 1)。
    """
    return x.t().contiguous().t()


# allow_in_graph tells torch.compile to treat this as an opaque operation —
# dynamo won't try to decompose it into smaller ops. See the module docstring
# for how this differs from torchao's tensor subclass approach.
@torch._dynamo.allow_in_graph  # 告知 torch.compile 将此视为不透明操作，不分解内部
class _Float8Matmul(torch.autograd.Function):
    """Linear 层的三个 FP8 GEMM 的自定义 autograd。
    forward 将 input 和 weight 量化为 FP8，保存量化张量 + scales 供 backward 使用。"""

    @staticmethod
    def forward(ctx, input_2d, weight):
        # Quantize both operands to e4m3 (higher precision format)
        input_fp8, input_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)
        ctx.save_for_backward(input_fp8, input_inv, weight_fp8, weight_inv)

        # output = input @ weight.T
        # input_fp8 is [B, K] contiguous = row-major (good for first arg)
        # weight_fp8 is [N, K] contiguous, so weight_fp8.t() is [K, N] with
        # strides (1, K) = column-major (good for second arg, no copy needed!)
        output = torch._scaled_mm(
            input_fp8,
            weight_fp8.t(),
            scale_a=input_inv,
            scale_b=weight_inv,
            out_dtype=input_2d.dtype,
            # use_fast_accum=True accumulates the dot products in lower precision.
            # Slightly less accurate but measurably faster. Standard practice for
            # the forward pass; we use False in backward for more precise gradients.
            use_fast_accum=True,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        in_fp8, in_inv, w_fp8, w_inv = ctx.saved_tensors

        # === GEMM 1: grad_input = grad_output @ weight ===
        # Shapes: [B, N] @ [N, K] -> [B, K]
        # Gradients use e5m2 (wider range), weights use e4m3 (higher precision)
        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)
        # go_fp8 is [B, N] contiguous = row-major, good for first arg
        # w_fp8 is [N, K] contiguous = row-major, need column-major for second arg
        w_col = _to_col_major(w_fp8)
        grad_input = torch._scaled_mm(
            go_fp8,
            w_col,
            scale_a=go_inv,
            scale_b=w_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        # === GEMM 2: grad_weight = grad_output.T @ input ===
        # Shapes: [N, B] @ [B, K] -> [N, K]
        # go_fp8 is [B, N] contiguous, we need go.T = [N, B] as first arg.
        # Transposing gives column-major, but first arg needs row-major,
        # so we must call .contiguous() to physically rearrange the memory.
        go_T = go_fp8.t().contiguous()  # [N, B] row-major
        in_col = _to_col_major(in_fp8)    # [B, K] column-major
        grad_weight = torch._scaled_mm(
            go_T,
            in_col,
            scale_a=go_inv,
            scale_b=in_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        return grad_input, grad_weight


class Float8Linear(nn.Linear):
    """nn.Linear 的直接替换，在 FP8 中执行计算。
    权重和偏置保持原始精度（如 fp32/bf16）。
    仅矩阵乘法通过 _Float8Matmul autograd 函数在 FP8 中执行。"""

    def forward(self, input):
        # 将 input 转换为 COMPUTE_DTYPE（通常为 bf16），因为 _scaled_mm 期望低精度输入
        input = input.to(COMPUTE_DTYPE)
        # _scaled_mm 仅支持 2D 张量，因此展平 batch 维度
        orig_shape = input.shape
        input_2d = input.reshape(-1, orig_shape[-1])
        output = _Float8Matmul.apply(input_2d, self.weight)
        output = output.reshape(*orig_shape[:-1], output.shape[-1])
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    @classmethod
    def from_float(cls, mod):
        """从 nn.Linear 创建 Float8Linear，共享相同的 weight 和 bias。
        使用 meta device 避免分配临时权重张量：在 meta 上创建模块外壳（仅形状/dtype），
        然后将 .weight 和 .bias 指向原始模块的参数。"""
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


class Float8LinearConfig:
    """与 torchao API 兼容的最小配置类。仅支持 tensorwise 方案。"""

    @staticmethod
    def from_recipe_name(recipe_name):
        if recipe_name != "tensorwise":
            raise ValueError(
                f"Only 'tensorwise' recipe is supported, got '{recipe_name}'. "
                f"Rowwise/axiswise recipes require the full torchao library."
            )
        return Float8LinearConfig()


def convert_to_float8_training(module, *, config=None, module_filter_fn=None):
    """在整个模块中将 nn.Linear 层替换为 Float8Linear。
    后序遍历模块树（子模块先于父模块），替换每个通过可选过滤器的 nn.Linear。
    新的 Float8Linear 共享原始权重和偏置张量 — 无复制，无额外内存。
    常用 filter：跳过维度不能被 16 整除的层（H100 上 FP8 matmul 的硬件要求）。"""
    def _convert(mod, prefix=""):
        for name, child in mod.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            _convert(child, fqn)
            if isinstance(child, nn.Linear) and not isinstance(child, Float8Linear):
                if module_filter_fn is None or module_filter_fn(child, fqn):
                    setattr(mod, name, Float8Linear.from_float(child))

    _convert(module)
    return module
