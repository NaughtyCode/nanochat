"""
高效的混合 AdamW/Muon 组合优化器。
通常将嵌入和标量参数交给 AdamW，矩阵参数交给 Muon。
提供两个版本：MuonAdamW（单 GPU）和 DistMuonAdamW（分布式训练）。

Muon（MomentUm Orthogonalized by Newton-schulz）通过正交化动量来加速训练，
Adapted from: https://github.com/KellerJordan/modded-nanogpt
Further contributions from @karpathy and @chrisjmccormick.
"""

import torch
import torch.distributed as dist
from torch import Tensor
from nanochat.common import COMPUTE_DTYPE

# -----------------------------------------------------------------------------
# AdamW 优化器 — 融合内核版本
# https://arxiv.org/abs/1711.05101
# -----------------------------------------------------------------------------

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,              # 参数张量
    grad: Tensor,           # 梯度（与 p 同形状）
    exp_avg: Tensor,        # 一阶动量（与 p 同形状）
    exp_avg_sq: Tensor,     # 二阶动量（与 p 同形状）
    step_t: Tensor,         # 0 维 CPU 张量 — 步数
    lr_t: Tensor,           # 0 维 CPU 张量 — 学习率
    beta1_t: Tensor,        # 0 维 CPU 张量 — beta1
    beta2_t: Tensor,        # 0 维 CPU 张量 — beta2
    eps_t: Tensor,          # 0 维 CPU 张量 — epsilon
    wd_t: Tensor,           # 0 维 CPU 张量 — weight decay
) -> None:
    """融合的 AdamW 步骤：weight_decay → momentum_update → bias_correction → param_update。
    全部在一个编译图中执行，消除 Python 操作间开销。
    所有超参数使用 0 维 CPU 张量，避免值变化时重新编译。"""
    # 解耦权重衰减（在更新前应用）
    p.mul_(1 - lr_t * wd_t)
    # 更新滑动平均（lerp_ 更简洁且融合效果好）
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    # 偏置校正
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    # 计算更新量并应用
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

# -----------------------------------------------------------------------------
# Muon 优化器 — 从 modded-nanogpt 改造简化而来
# https://github.com/KellerJordan/modded-nanogpt
#
# 背景：使用 Newton-Schulz 迭代计算梯度矩阵 G 的"零次方"/正交化。
# 我们采用五次迭代，其系数选择为最大化零点的斜率。经验表明，即使迭代
# 不再在区间内处处收敛到 1，继续增大零点斜率也是有效的。因此该迭代
# 不产生 UV^T，而是产生类似 US'V^T 的结果，其中 S' 的对角元素约为
# Uniform(0.5, 1.5)，这对模型性能几乎没有影响（相对于 SVD 的精确结果）。
#
# 此处使用 Polar Express 符号方法作为 Newton-Schulz 迭代的替代方案，
# 具有更好的收敛性：
# https://arxiv.org/pdf/2505.16932
# 作者: Noah Amsel, David Persson, Christopher Musco, Robert M. Gower
#
# NorMuon 方差缩减：逐神经元/逐列自适应学习率，
# 在正交化后归一化更新尺度（Muon 的输出在神经元间有非均匀的尺度）。
# https://arxiv.org/pdf/2510.05491
#
# nanochat 实现的改进：
# - 使用更简单、更通用的参数分组和堆叠方法
# - 使用单一融合内核：momentum → polar_express → variance_reduction → update
# - 不做模型架构假设（例如不假设注意力权重已融合为 QKVO 格式）
# -----------------------------------------------------------------------------

# Polar Express 系数（num_iters=5, safety_factor=2e-2, cushion=2）
# 来源：https://arxiv.org/pdf/2505.16932
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,          # 堆叠的梯度
    stacked_params: Tensor,         # 堆叠的参数
    momentum_buffer: Tensor,        # 一阶动量缓冲区
    second_momentum_buffer: Tensor, # 分解的二阶动量（按行或按列）
    momentum_t: Tensor,             # 0 维 CPU 张量 — 动量系数
    lr_t: Tensor,                   # 0 维 CPU 张量 — 学习率
    wd_t: Tensor,                   # 0 维 CPU 张量 — weight decay
    beta2_t: Tensor,                # 0 维 CPU 张量 — 二阶动量 beta2
    ns_steps: int,                  # Newton-Schulz/Polar Express 迭代次数（通常为 5）
    red_dim: int,                   # 方差缩减维度（-1=按行, -2=按列）
) -> None:
    """融合的 Muon 步骤：momentum → polar_express → variance_reduction → cautious_update。
    全部在一个编译图中执行，消除 Python 操作间开销。"""

    # Nesterov 动量：先用当前动量预测下一步位置，再在该位置评估梯度
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar Express 正交化
    # 如果可用则转换为 bf16 加速；fp16 因指数范围有限在此不稳定，跳过转换
    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)  # 归一化到接近单位范数
    if g.size(-2) > g.size(-1):
        # 高矩阵（行 > 列）：使用转置公式以减小中间矩阵大小
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        # 宽矩阵（列 >= 行）：使用原始公式
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    # NorMuon 方差缩减：逐神经元自适应学习率
    # 归一化正交化后的更新尺度，防止某些神经元更新过大
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # 谨慎权重衰减 + 参数更新
    # "谨慎"：只在更新方向与参数符号一致的位置应用权重衰减
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

# -----------------------------------------------------------------------------
# 单 GPU 版本的 MuonAdamW 优化器
# 主要用于参考、调试和测试
# -----------------------------------------------------------------------------

class MuonAdamW(torch.optim.Optimizer):
    """组合优化器：Muon 用于 2D 矩阵参数，AdamW 用于其他参数（单 GPU 版本）。

    AdamW — 标准 AdamW 优化器步骤（融合内核加速）。

    Muon — MomentUm Orthogonalized by Newton-schulz
    https://kellerjordan.github.io/posts/muon/

    Muon 内部运行标准 SGD 动量，然后执行正交化后处理步骤，将每个 2D 参数的更新
    替换为最近的正交矩阵。为高效地正交化每个更新，使用 Newton-Schulz 迭代，
    其优势是可以在 GPU 上以 bfloat16 稳定运行。

    注意事项：
    - Muon 不应用于嵌入层、最终全连接层或任何 {0,1}-D 参数
    - 嵌入层和标量参数应使用 AdamW
    - 对 4D 卷积滤波器，将其后 3 维展平即可使用 Muon

    参数：param_groups: 字典列表，每个包含：
        - 'params': 参数列表
        - 'kind': 'adamw' 或 'muon'
        - AdamW 组: 'lr', 'betas', 'eps', 'weight_decay'
        - Muon 组: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # 0 维 CPU 张量：避免值变化时 torch.compile 重新编译
        # AdamW 张量
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # Muon 张量
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group: dict) -> None:
        """对组中每个参数执行 AdamW 更新。
        延迟初始化状态，填充所有 0 维张量，调用融合内核。"""
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]

            # State init
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            state['step'] += 1

            # Fill 0-D tensors with current values
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])

            # Fused update: weight_decay -> momentum -> bias_correction -> param_update
            adamw_step_fused(
                p, grad, exp_avg, exp_avg_sq,
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

    def _step_muon(self, group: dict) -> None:
        """对组中所有参数执行 Muon 更新（堆叠以提高效率）。
        延迟初始化状态，填充所有 0 维张量，调用融合内核。"""
        params: list[Tensor] = group['params']
        if not params:
            return

        # Get or create group-level buffers (stored in first param's state for convenience)
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        # Momentum for every individual parameter
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        momentum_buffer = state["momentum_buffer"]

        # Second momentum buffer is factored, either per-row or per-column
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        second_momentum_buffer = state["second_momentum_buffer"]
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Stack grads and params (NOTE: this assumes all params have the same shape)
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        # Fill all the 0-D tensors with current values
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        # Single fused kernel: momentum -> polar_express -> variance_reduction -> update
        muon_step_fused(
            stacked_grads,
            stacked_params,
            momentum_buffer,
            second_momentum_buffer,
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )

        # Copy back to original params
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        """对所有参数组执行一步优化。按 kind 分发到 AdamW 或 Muon 更新。"""
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

# -----------------------------------------------------------------------------
# 分布式版本的 MuonAdamW 优化器
# 用于多 GPU 训练，采用 ZeRO-2 风格的优化器状态分片
# -----------------------------------------------------------------------------

class DistMuonAdamW(torch.optim.Optimizer):
    """组合分布式优化器：Muon 用于 2D 矩阵参数，AdamW 用于其他参数。

    算法细节见 MuonAdamW。此类添加了分布式通信以实现多 GPU 训练（不依赖 PyTorch DDP）。

    设计目标：
    - 通信与计算重叠（异步操作）
    - 通过跨 rank 分片优化器状态最小化内存（ZeRO-2 风格）
    - 尽可能将小张量批量合并为单个通信操作

    通信模式（3 阶段异步）：
    使用 3 阶段结构最大化通信与计算的重叠：

        阶段 1: 启动所有异步 reduce 操作
            - 发起所有 reduce_scatter/all_reduce 操作
            - 不等待 — 让它们在后台运行

        阶段 2: 等待 reduce 完成 → 计算更新 → 启动 gather
            - 对每组：等待其 reduce，计算更新，启动 gather
            - 按顺序处理各组，早期 gather 运行期间后期计算同步进行

        阶段 3: 等待 gather 完成，复制回去
            - 等待所有 gather 完成
            - 将更新后的参数复制回原始张量（仅 Muon 需要）

    AdamW 通信（ZeRO-2 风格）：
    - 小参数（<1024 元素）：all_reduce 梯度，在每个 rank 上更新完整参数。
      优化器状态被复制，但这些参数很微小（标量、偏置）。
    - 大参数：reduce_scatter 梯度使每个 rank 获得 1/N 的梯度，
      仅更新该切片，然后 all_gather 更新后的切片。
      优化器状态（exp_avg, exp_avg_sq）被分片 — 每个 rank 只存储其切片的状态。
      要求 param.shape[0] 能被 world_size 整除。

    Muon 通信（堆叠 + 分块）：
    - 同一 Muon 组中的所有参数必须具有相同形状（调用者负责）。
    - 将所有 K 个参数堆叠为单个 (K, *shape) 张量以实现高效通信。
    - 将 K 个参数分配到 N 个 rank：每个 rank "拥有" ceil(K/N) 个参数。
    - reduce_scatter 堆叠的梯度，每个 rank 获得其块。
    - 每个 rank 仅为自己拥有的参数计算 Muon 更新。
    - all_gather 更新后的参数回所有 rank。
    - 优化器状态（momentum_buffer, second_momentum_buffer）按块分片。
    - 填充：如果 K 不能被整除，零填充到 ceil(K/N) * N 用于通信，复制时忽略填充。

    缓冲区重用：
    - 对 Muon，我们分配 stacked_grads 作为 reduce_scatter 输入，然后重用
      相同缓冲区作为 all_gather 的输出（stacked_params），节省内存。
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _reduce_adamw(self, group: dict, world_size: int) -> dict:
        """阶段 1：为 AdamW 组启动异步 reduce 操作。
        小参数使用 all_reduce，大参数使用 reduce_scatter（ZeRO-2 分片）。"""
        param_infos = {}
        for p in group['params']:
            grad = p.grad
            if p.numel() < 1024:
                # Small params: all_reduce (no scatter/gather needed)
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                # Large params: reduce_scatter
                assert grad.shape[0] % world_size == 0, f"AdamW reduce_scatter requires shape[0] ({grad.shape[0]}) divisible by world_size ({world_size})"
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=param_infos)

    def _reduce_muon(self, group: dict, world_size: int) -> dict:
        """阶段 1：为 Muon 组启动异步 reduce_scatter 操作。
        所有同形状参数堆叠后进行分块通信。"""
        params = group['params']
        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # Stack grads and zero-pad to padded_num_params
        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(grad_stack)
        if len(params) < padded_num_params:
            stacked_grads[len(params):].zero_()

        # Reduce_scatter to get this rank's chunk
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()

        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _compute_adamw(self, group: dict, info: dict, gather_list: list, rank: int, world_size: int) -> None:
        """阶段 2：等待 reduce 完成，计算 AdamW 更新，为大参数启动 all_gather。"""
        param_infos = info['param_infos']
        for p in group['params']:
            pinfo = param_infos[p]
            pinfo['future'].wait()
            grad_slice = pinfo['grad_slice']
            state = self.state[p]

            # For small params, operate on full param; for large, operate on slice
            if pinfo['is_small']:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]

            # State init
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
            state['step'] += 1

            # Fill 0-D tensors and run fused kernel
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(
                p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

            # Large params need all_gather
            if not pinfo['is_small']:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append(dict(future=future, params=None))

    def _compute_muon(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        """阶段 2：等待 reduce 完成，计算 Muon 更新，启动 all_gather。"""
        info['future'].wait()
        params = group['params']
        chunk_size = info['chunk_size']
        grad_chunk = info['grad_chunk']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # How many params does this rank own?
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        # Get or create group-level state
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Build output buffer for all_gather
        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            # Fill 0-D tensors and run fused kernel
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
            self._muon_wd_t.fill_(group["weight_decay"])
            muon_step_fused(
                grad_chunk[:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                group["ns_steps"], red_dim,
            )
            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        # Reuse stacked_grads buffer for all_gather output
        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    def _finish_gathers(self, gather_list: list) -> None:
        """阶段 3：等待所有 gather 完成，将 Muon 参数复制回原始张量。"""
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
                # Muon: copy from stacked buffer back to individual params
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))

    @torch.no_grad()
    def step(self):
        """分布式优化步骤（3 阶段异步流水线）。"""
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # 阶段 1：启动所有异步 reduce 操作（不等待，让它们在后台运行）
        reduce_infos: list[dict] = []
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group['kind'] == 'muon':
                reduce_infos.append(self._reduce_muon(group, world_size))
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # 阶段 2：等待 reduce → 计算更新 → 启动 gather
        gather_list: list[dict] = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group['kind'] == 'adamw':
                self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group['kind'] == 'muon':
                self._compute_muon(group, info, gather_list, rank)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # 阶段 3：等待所有 gather 完成并复制回去
        self._finish_gathers(gather_list)
